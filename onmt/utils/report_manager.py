""" Report manager utility """
from __future__ import print_function
import os
import time
from datetime import datetime

import onmt

from onmt.utils.logging import logger


def build_report_manager(opt, gpu_rank):
    if opt.tensorboard and gpu_rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_log_dir = opt.tensorboard_log_dir

        if not opt.train_from:
            tensorboard_log_dir += datetime.now().strftime("/%b-%d_%H-%M-%S")

        writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
    else:
        writer = None

    record_dir = None
    if gpu_rank == 0:
        record_dir = opt.save_dir

    report_mgr = ReportMgr(opt.report_every, num_experts=opt.num_experts, 
                           start_time=-1, tensorboard_writer=writer,
                           record_dir=record_dir)
    return report_mgr


class ReportMgrBase(object):
    """
    Report Manager Base class
    Inherited classes should override:
        * `_report_training`
        * `_report_step`
    """

    def __init__(self, report_every, num_experts=1, start_time=-1.):
        """
        Args:
            report_every(int): Report status every this many sentences
            start_time(float): manually set report start time. Negative values
                means that you will need to set it later or use `start()`
        """
        self.report_every = report_every
        self.start_time = start_time
        self.num_experts = num_experts

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def report_training(self, step, num_steps, learning_rate,
                        report_stats, multigpu=False):
        """
        This is the user-defined batch-level traing progress
        report function.

        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(Statistics): old Statistics instance.
        Returns:
            report_stats(Statistics): updated Statistics instance.
        """
        if self.start_time < 0:
            raise ValueError("""ReportMgr needs to be started
                                (set 'start_time' or use 'start()'""")

        if step % self.report_every == 0:
            if multigpu:
                report_stats = \
                    onmt.utils.Statistics.all_gather_stats(report_stats)
            self._report_training(
                step, num_steps, learning_rate, report_stats)
            return onmt.utils.Statistics(self.num_experts)
        else:
            return report_stats

    def _report_training(self, *args, **kwargs):
        """ To be overridden """
        raise NotImplementedError()

    def report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        Report stats of a step

        Args:
            train_stats(Statistics): training stats
            valid_stats(Statistics): validation stats
            lr(float): current learning rate
        """
        self._report_step(
            lr, step, train_stats=train_stats, valid_stats=valid_stats)
        self._record_step(
            lr, step, train_stats=train_stats, valid_stats=valid_stats)

    def _report_step(self, *args, **kwargs):
        raise NotImplementedError()

    def _record_step(self, *args, **kwargs):
        raise NotImplementedError()


class ReportMgr(ReportMgrBase):
    def __init__(self, report_every, num_experts=1, start_time=-1., 
                 tensorboard_writer=None, record_dir=None):
        """
        A report manager that writes statistics on standard output as well as
        (optionally) TensorBoard

        Args:
            report_every(int): Report status every this many sentences
            tensorboard_writer(:obj:`tensorboard.SummaryWriter`):
                The TensorBoard Summary writer to use or None
        """
        super(ReportMgr, self).__init__(report_every, num_experts, start_time)
        self.tensorboard_writer = tensorboard_writer
        self.record_train = self.record_valid = None
        if record_dir is not None:
            self.record_train = os.path.join(record_dir, 'record_train.csv')
            self.record_valid = os.path.join(record_dir, 'record_valid.csv')

    def maybe_log_tensorboard(self, stats, prefix, learning_rate, step):
        if self.tensorboard_writer is not None:
            stats.log_tensorboard(
                prefix, self.tensorboard_writer, learning_rate, step)

    def _report_training(self, step, num_steps, learning_rate,
                         report_stats):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        report_stats.output(step, num_steps,
                            learning_rate, self.start_time)

        self.maybe_log_tensorboard(report_stats,
                                   "progress",
                                   learning_rate,
                                   step)
        self._record_step(learning_rate, step, train_stats=report_stats)
        report_stats = onmt.utils.Statistics(self.num_experts)

        return report_stats

    def _report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        if train_stats is not None:
            self.log('Train perplexity: %g' % train_stats.ppl())
            self.log('Train accuracy: %g' % train_stats.accuracy())

            self.maybe_log_tensorboard(train_stats,
                                       "train",
                                       lr,
                                       step)

        if valid_stats is not None:
            self.log('Validation perplexity: %g; %g' 
                     % (valid_stats.ppl('x2y'), 
                        valid_stats.ppl('y2x')))
            self.log('Validation accuracy: %g; %g' 
                     % (valid_stats.accuracy('x2y'), 
                        valid_stats.accuracy('y2x')))
            self.maybe_log_tensorboard(valid_stats,
                                       "valid",
                                       lr,
                                       step)

    def _record_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        if train_stats is not None and self.record_train is not None:
            train_stats.log_records(step, lr, self.record_train)
        if valid_stats is not None and self.record_valid is not None: 
            valid_stats.log_records(step, lr, self.record_valid)