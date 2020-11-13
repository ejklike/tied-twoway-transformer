""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys
import numpy as np

from onmt.utils.logging import logger


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, num_experts, loss=0,
                 loss_x2y=0, loss_y2x=0, 
                 n_words_x2y=0, n_words_y2x=0, 
                 n_correct_x2y=0, n_correct_y2x=0):
        self.num_experts = num_experts
        self.loss = loss

        self.loss_x2y = loss_x2y
        self.loss_y2x = loss_y2x
        self.n_words_x2y = n_words_x2y
        self.n_words_y2x = n_words_y2x
        self.n_correct_x2y = n_correct_x2y
        self.n_correct_y2x = n_correct_y2x
        self.start_time = time.time()

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        from torch.distributed import get_rank
        from onmt.utils.distributed import all_gather_list

        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat)
        return our_stats

    def update(self, stat):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object

        """
        assert self.num_experts == stat.num_experts, (self.num_experts,  stat.num_experts)
        self.loss += stat.loss

        self.loss_x2y += stat.loss_x2y
        self.loss_y2x += stat.loss_y2x
        self.n_words_x2y += stat.n_words_x2y
        self.n_words_y2x += stat.n_words_y2x
        self.n_correct_x2y += stat.n_correct_x2y
        self.n_correct_y2x += stat.n_correct_y2x

    def accuracy(self, side):
        """ compute accuracy """
        if side == 'x2y':
            return (100 * (self.n_correct_x2y / self.n_words_x2y)
                    if self.n_words_x2y > 0 else 0)
        else:
            return (100 * (self.n_correct_y2x / self.n_words_y2x)
                    if self.n_words_y2x > 0 else 0)

    def xent(self, side):
        """ compute cross entropy """
        if side == 'x2y':
            return (self.loss_x2y / self.n_words_x2y 
                    if self.n_words_x2y > 0 else 0)
        else:
            return (self.loss_y2x / self.n_words_y2x
                    if self.n_words_y2x > 0 else 0)

    def ppl(self, side):
        """ compute perplexity """
        if side == 'x2y':
            return (math.exp(min(self.loss_x2y / self.n_words_x2y, 100))
                    if self.n_words_x2y > 0 else 0)
        else:
            return (math.exp(min(self.loss_y2x / self.n_words_y2x, 100))
                    if self.n_words_y2x > 0 else 0)

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        logger.info(
            "Step %s; lr: %7.10f; %6.0f sec;"
            % (step_fmt, learning_rate, time.time() - start))
        logger.info(
            "[FWD] acc: %3.2f; ppl: %3.2f; xent: %3.2f; %3.0f tok/s"
            % (self.accuracy('x2y'), self.ppl('x2y'), self.xent('x2y'),
               (self.n_words_x2y) / (t + 1e-5)))
        logger.info(
            "[BWD] acc: %3.2f; ppl: %3.2f; xent: %3.2f; %3.0f tok/s"
            % (self.accuracy('y2x'), self.ppl('y2x'), self.xent('y2x'),
               (self.n_words_y2x) / (t + 1e-5)))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent_x2y", self.xent('x2y'), step)
        writer.add_scalar(prefix + "/ppl_x2y", self.ppl('x2y'), step)
        writer.add_scalar(prefix + "/accuracy_x2y", self.accuracy('x2y'), step)
        writer.add_scalar(prefix + "/tgtper_x2y", self.n_words_x2y / t, step)

        writer.add_scalar(prefix + "/xent_y2x", self.xent('y2x'), step)
        writer.add_scalar(prefix + "/ppl_y2x", self.ppl('y2x'), step)
        writer.add_scalar(prefix + "/accuracy_y2x", self.accuracy('y2x'), step)
        writer.add_scalar(prefix + "/tgtper_y2x", self.n_words_y2x / t, step)
        
        writer.add_scalar(prefix + "/lr", learning_rate, step)

    def log_records(self, step, lr, record_to=None):
        assert record_to is not None
        with open(record_to, 'a+') as f:
            line = ('%d,%g,%g,%g,%g,%g,%g\n' % (
                    step, self.elapsed_time(), lr, 
                    self.accuracy('x2y'), 
                    self.accuracy('y2x'),
                    self.ppl('x2y'),
                    self.ppl('y2x')))
            f.write(line)