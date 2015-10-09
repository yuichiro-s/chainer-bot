#!/usr/bin/env python

from bot import vocab

import sys
from collections import defaultdict


def main(args):
    # word frequency
    freq = defaultdict(int)

    with open(args.corpus) as f:
        for line in f:
            line = line.strip().decode('utf-8')

            # segment sentence into words
            if args.char:
                ws = line
            else:
                ws = line.split()

            for w in ws:
                freq[w] += 1

    # register words
    voc = vocab.Vocab()
    for k, v in sorted(freq.items(), key=lambda (k, v): -v):
        if voc.size() >= args.size:
            break
        print >> sys.stderr, 'Adding word: {}\tcnt: {}'.format(k.encode('utf-8'), v)
        voc.add_word(k)

    # save vocabulary
    print >> sys.stderr, 'Writing vocabulary to {}'.format(args.vocab)
    voc.save(args.vocab)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create vocabulary from parallel corpus.')

    # NN architecture
    parser.add_argument('corpus', help='raw corpus file')
    parser.add_argument('vocab', help='destination of vocabulary')
    parser.add_argument('size', type=int, help='vocabulary size')
    parser.add_argument('--char', action='store_true', default=False, help='segment sentences character-wise')

    main(parser.parse_args())
