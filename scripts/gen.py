#!/usr/bin/env python

from bot import util
from bot import vocab
from bot import gen
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

    # arguments for generation method
    kwargs = vars(args)
    kwargs.pop('model')

    while True:
        try:
            # get input
            user_input = raw_input('> ').decode('utf-8')

            prefix = None
            if user_input.startswith(u':'):
                # prefix mode
                prefix = user_input[1:]
                user_input = u''

            # separate into words
            # currently all characters are treated as one word
            input_words = user_input

            sen = gen.generate(input_words, s2s, voc, prefix=prefix, **kwargs)

            print sen

        except Exception as e:
            print >> sys.stderr, 'Error: ' + str(e)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate sentences from model interactively.')

    parser.add_argument('model', help='destination of model')
    parser.add_argument('--exp', type=float, default=3, help='adjust output distribution by exponentiating it by this number')
    parser.add_argument('--min-len', type=int, default=0, help='minimum length of output seuqnece')
    parser.add_argument('--max-len', type=int, default=50, help='maximum length of output seuqnece')
    parser.add_argument('--no-unk', action='store_true', default=False, help='don\'t generate <UNK>')
    parser.add_argument('--exclude', type=int, nargs='+', help='IDs never to generate')
    parser.add_argument('--exclude-first', type=int, nargs='+', help='IDs not to generate as the first word')

    main(parser.parse_args())
