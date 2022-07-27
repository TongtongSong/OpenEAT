#!/usr/bin/env python3
# encoding: utf-8

from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import argparse
import codecs
from distutils.util import strtobool
from io import open
import logging
import sys

PY2 = sys.version_info[0] == 2
sys.stdin = codecs.getreader('utf-8')(sys.stdin if PY2 else sys.stdin.buffer)
sys.stdout = codecs.getwriter('utf-8')(
    sys.stdout if PY2 else sys.stdout.buffer)


# Special types:
def shape(x):
    """Change str to List[int]

    >>> shape('3,5')
    [3, 5]
    >>> shape(' [3, 5] ')
    [3, 5]

    """

    # x: ' [3, 5] ' -> '3, 5'
    x = x.strip()
    if x[0] == '[':
        x = x[1:]
    if x[-1] == ']':
        x = x[:-1]

    return list(map(int, x.split(',')))


def get_parser():
    parser = argparse.ArgumentParser(
        description='Given each file paths with such format as '
        '<key>:<file>:<type>. type> can be omitted and the default '
        'is "str". e.g. {} '
        '--input-scps feat:data/feats.scp shape:data/utt2feat_shape:shape '
        '--input-scps feat:data/feats2.scp shape:data/utt2feat2_shape:shape '
        '--output-scps text:data/text shape:data/utt2text_shape:shape '
        '--scps utt2spk:data/utt2spk'.format(sys.argv[0]),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-scps',
                        type=str,
                        nargs='*',
                        action='append',
                        default=[],
                        help='files for the inputs')
    parser.add_argument('--output-scps',
                        type=str,
                        nargs='*',
                        action='append',
                        default=[],
                        help='files for the outputs')
    parser.add_argument('--scps',
                        type=str,
                        nargs='+',
                        default=[],
                        help='The files except for the input and outputs')
    parser.add_argument('--verbose',
                        '-V',
                        default=1,
                        type=int,
                        help='Verbose option')
    parser.add_argument('--allow-one-column',
                        type=strtobool,
                        default=False,
                        help='Allow one column in input scp files. '
                        'In this case, the value will be empty string.')
    parser.add_argument('--out',
                        '-O',
                        type=str,
                        help='The output filename. '
                        'If omitted, then output to sys.stdout')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.scps = [args.scps]

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)

    inputs = {}
    assert (len(args.input_scps) == 1)
    for f in args.input_scps[0]:
        arr = f.strip().split(':')
        inputs[arr[0]] = arr[1]
    assert ('feat' in inputs)
    assert ('shape' in inputs)

    outputs = {}
    assert (len(args.output_scps) == 1)
    for f in args.output_scps[0]:
        arr = f.strip().split(':')
        outputs[arr[0]] = arr[1]
    assert ('text' in outputs)

    files = [
        inputs['feat'], inputs['shape'], 
        outputs['text']
    ]
    fields = ['feat', 'feat_shape', 'text']
    fids = [open(f, 'r', encoding='utf-8') for f in files]

    # the file with the minimum number of lines in the first

    if args.out is None:
        out = sys.stdout
    else:
        out = open(args.out, 'w', encoding='utf-8')

    # merage by least file's uttid
    # data={}
    # least_num_idx=-1
    # least_num=float('inf')
    # for i, fid in enumerate(fids):
    #     data[i] = {}
    #     lines = fid.readlines()
    #     if len(lines) < least_num:
    #         least_num = len(lines)
    #         least_num_idx = i
    #     for line in lines:
    #         arr = line.strip().split()
    #         content = ' '.join(arr[1:])
    #         data[i][arr[0]] = content
    # for idx, key in enumerate(data[least_num_idx].keys()):
    #     out.write('utt:{}'.format(key))
    #     for i, t in enumerate(fields):
    #         out.write('\t')
    #         out.write('{}:{}'.format(fields[i], data[i][key]))
    #     out.write('\n')

    # merage by line
    done = False
    while not done:
        for i, fid in enumerate(fids):
            line = fid.readline()
            if line == '':
                done = True
                break
            arr = line.strip().split()
            content = ' '.join(arr[1:])
            if i == 0:
                out.write('utt:{}'.format(arr[0]))
            out.write('\t')
            out.write('{}:{}'.format(fields[i], content))
        out.write('\n')

    for f in fids:
        f.close()
    if args.out is not None:
        out.close()
