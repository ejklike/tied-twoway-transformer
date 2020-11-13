#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from myutils import evaluate


def main():
    parser = argparse.ArgumentParser(description='evaluate.py')

    parser.add_argument('-beam_size', type=int, default=10,
                       help='Beam size')
    parser.add_argument('-n_best', type=int, default=10,
                       help='n_best')
    parser.add_argument('-output', type=str, required=True,
                       help="Path to file containing the translation outputs")
    parser.add_argument('-target', type=str, required=True,
                       help="Path to file containing scoring results")
    opt = parser.parse_args()

    print(opt.target)
    print(opt.output)
    evaluate(opt.output, opt.target, n_best=opt.n_best, 
             save_score_to=opt.output + '.score', save_rawdata=True)


if __name__ == "__main__":
    main()
