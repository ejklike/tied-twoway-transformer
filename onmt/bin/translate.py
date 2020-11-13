#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import codecs
from multiprocessing import Process, Queue

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

from myutils import (canonicalize_smitxt, 
                     linecount, 
                     tile_lines_n_times,
                     combine_N_translations,
                     evaluate)


def translate(opt, expert_id):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    opt.gpu = expert_id % opt.ngpu
    logger.info('OPT_GPU: %d' % opt.gpu)
    translator = build_translator(opt, logger=logger,
                                  report_score=True, 
                                  log_score=True)
    translator.expert_id = expert_id

    desired_output_length = linecount(opt.src) * opt.n_best
    logger.info(opt.src)
    logger.info(opt.tgt)

    # tiled src file
    opt.tiled_src = opt.src + '.x%d' % opt.n_best
    tile_lines_n_times(opt.src, opt.tiled_src, n=opt.n_best)

    logger.info("=== FWD ===")
    src_path = opt.src
    tgt_path = None #opt.tgt
    out_path = opt.output + '/fwd_out%d.txt' % expert_id
    out_can_path = opt.output + '/fwd_out%d_can.txt' % expert_id

    if linecount(out_path) == desired_output_length:
        logger.info("Already translated. Pass.")
    else:
        # data preparation
        src_shards = split_corpus(src_path, opt.shard_size)
        tgt_shards = split_corpus(tgt_path, opt.shard_size)
        shard_pairs = zip(src_shards, tgt_shards)
        # translate
        translator.out_file = codecs.open(out_path, 'w+', 'utf-8')
        for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
            logger.info("Translating shard %d." % i)
            translator.translate(
                direction='x2y',
                src=src_shard,
                tgt=tgt_shard,
                src_dir=opt.src_dir,
                batch_size=opt.batch_size,
                batch_type=opt.batch_type,
                attn_debug=opt.attn_debug,
                align_debug=opt.align_debug)
        # canonicalize
        canonicalize_smitxt(out_path, out_can_path, 
                            remove_score=True)

    logger.info("=== BWD ===")
    translator.beam_size = 1
    translator.n_best = 1

    src_path = opt.output + '/fwd_out%d_can.txt' % expert_id
    tgt_path = opt.tiled_src
    out_path = opt.output + '/bwd_out%d.txt' % expert_id

    if linecount(out_path) == desired_output_length:
        logger.info("Already translated. Pass.")
    else:
        # data preparation
        src_shards = split_corpus(src_path, opt.shard_size)
        tgt_shards = split_corpus(tgt_path, opt.shard_size)
        shard_pairs = zip(src_shards, tgt_shards)
        # translate
        translator.out_file = codecs.open(out_path, 'w+', 'utf-8')
        for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
            logger.info("Translating shard %d." % i)
            translator.translate(
                direction='y2x',
                src=src_shard,
                tgt=tgt_shard,
                src_dir=opt.src_dir,
                batch_size=opt.batch_size,
                batch_type=opt.batch_type,
                attn_debug=opt.attn_debug,
                align_debug=opt.align_debug,
                only_gold_score=True) # for speed


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()

    n_best = opt.n_best
    opt.n_best = opt.beam_size

    if opt.num_experts > opt.ngpu:
        procs = []
        for i in range(opt.num_experts):
            p = Process(target=translate, args=(opt, i))
            procs.append(p)

        for i in range(0, opt.num_experts, opt.ngpu):
            print(i)
            for p in procs[i:i+opt.ngpu]:
                p.start()
            for p in procs[i:i+opt.ngpu]:
                p.join()

    else:
        procs = []
        for i in range(opt.num_experts):
            p = Process(target=translate, args=(opt, i))
            procs.append(p)
            p.start()

        for p in procs:
            p.join()

    input_format = opt.output + '/fwd_out%d.txt'
    bwd_input_format = opt.output + '/bwd_out%d.txt'

    # combine K outputs
    print("=== Combine %d outputs ===" % opt.num_experts)
    pred_output_path = opt.output + '/pred_cycle_lp%d.txt' % opt.num_experts
    combine_N_translations(input_format, pred_output_path, 
                            opt.num_experts, opt.beam_size, n_best, 
                            bwd_input_format=bwd_input_format)
    if opt.tgt:
        evaluate(pred_output_path, opt.tgt, n_best=n_best, 
                save_score_to=pred_output_path + '.score', save_rawdata=True)


if __name__ == '__main__':
    main()
