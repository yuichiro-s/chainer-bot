#!/usr/bin/env python

from bot import vocab

import sys


def main(args):
    print >> sys.stderr, 'Loading vocabulary from {}'.format(args.vocab)
    voc = vocab.Vocab.load(args.vocab)
    print >> sys.stderr, 'Vocabulary size: {}'.format(voc.size())
    assert voc.get_id(vocab.EOS) == 0
    assert voc.get_id(vocab.UNK) == 1

    src = None
    print >> sys.stderr, 'Processing {}'.format(args.corpus)
    with open(args.corpus) as f, open(args.dest, 'w') as f_out:
        for line in f:
            line = line.rstrip().decode('utf-8')
            if len(line) > 0:
                # segment sentence into words
                if args.char:
                    ws = line
                else:
                    ws = line.split()
                ids_str = ' '.join(map(lambda w: str(voc.get_id(w)), ws))

                if src is None:
                    src = ids_str
                else:
                    trg = ids_str

                    f_out.write(src)
                    f_out.write('\t')
                    f_out.write(trg)
                    f_out.write('\n')

                    if args.dialog:
                        src = trg
                    else:
                        src = None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create vocabulary from parallel corpus.')

    # NN architecture
    parser.add_argument('corpus', help='corpus file')
    parser.add_argument('dest', help='destination of converted corpus')
    parser.add_argument('vocab', help='vocabulary file')
    parser.add_argument('--char', action='store_true', default=False, help='segment sentences character-wise')
    parser.add_argument('--dialog', action='store_true', default=False, help='use target sequence as source sequence for next sample')

    main(parser.parse_args())
