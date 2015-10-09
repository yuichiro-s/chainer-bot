#!/usr/bin/env python

from bot import util
from bot import vocab
from bot import seq2seq

import sys
import os


def main(args):
    # load model
    s2s = seq2seq.Seq2Seq.load(args.model)

    # load vocabulary
    dir_path = os.path.dirname(args.model)
    vocab_path = os.path.join(dir_path, 'vocab')
    voc = vocab.Vocab.load(vocab_path)

    while True:
        # get input
        user_input = raw_input('> ').decode('utf-8')

        # separate into words
        # currently all characters are treated as one word
        input_words = user_input

        # convert to IDs
        xs = []
        for w in input_words:
            w_id = voc.get_id(w)
            x = util.id2var(w_id, train=False)
            xs.append(x)

        # generate ID sequence
        w_ids = s2s.generate(xs, max_len=args.max_len, exp=args.exp)

        # convert to words
        ws = []
        for w_id in w_ids:
            w = voc.get_word(w_id)
            ws.append(w)

        # print words
        sen = u''.join(ws)
        print sen


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate sentences from model interactively.')

    parser.add_argument('model', help='destination of model')
    parser.add_argument('--exp', type=int, default=3, help='adjust output distribution by exponentiating it by this number')
    parser.add_argument('--max-len', type=int, default=50, help='maximum length of output seuqnece')

    main(parser.parse_args())
