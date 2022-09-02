#!/usr/bin/env python3

import argparse
import codecs
import sys

def get_parser():
    parser = argparse.ArgumentParser(
        description='convert raw text to vocabulary',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--skip-ncols',
                        '-s',
                        default=0,
                        type=int,
                        help='skip first n columns')
    parser.add_argument('--bpe-model',
                        '-m',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('text',
                        type=str,
                        default=False,
                        nargs='?',
                        help='input text')
    return parser


def main():
    char_dict = []
    parser = get_parser()
    args = parser.parse_args()

    if args.bpe_model is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(args.bpe_model)
    with codecs.open(args.text, encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().split()
            a_chars = []
            for w in line:
                if args.bpe_model:
                    for k in sp.encode_as_pieces(w):
                        a_chars.append(k)
                else:
                    for k in w:
                        a_chars.append(k)
            for a in a_chars:
                if a not in char_dict:
                    char_dict.append(a)
            line = f.readline()
    for c in char_dict:
        print(str(c))
if __name__ == '__main__':
    main()
