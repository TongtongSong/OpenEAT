# encoding=utf-8
import sys
import os
path = os.path.abspath(__file__)
path1 = os.path.split(path)[0]
path2 = os.path.split(path1)[0]
path3 = os.path.split(path2)[0]
sys.path.append(path3)

import argparse
import re

from openeat.dataset.text_processor import _remove_punctuation, _tokenizer

def get_parser():
    parser = argparse.ArgumentParser(
        description='convert raw text to tokenized text',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bpe_model',
                        '-b',
                        type=str,
                        default=None,
                        help="""Bpe model to tokenize English words""")
    parser.add_argument('text',
                        type=str,
                        default=False,
                        nargs='?',
                        help='input text')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.bpe_model is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(args.bpe_model)
    else:
        sp = None
    with open(args.text, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            uttid = line[0]
            text = ' '.join(line[1:])
            text = _remove_punctuation(text)
            text = text.replace('UNK','*').replace('unk','*')
            tokens = _tokenizer(text, sp)
            print(uttid+' '+' '.join(tokens))
            
if __name__ == '__main__':
    main()
