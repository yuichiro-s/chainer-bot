#!/usr/bin/env python

import sys

def main(args):
    src = None
    for line in sys.stdin:
        line = line.rstrip()
        if len(line) > 0:
            if src is None:
                src = line
            else:
                trg = line
                src, trg
    args.


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create vocabulary from parallel corpus. STDIN: corpus STDOUT: vocabulary')

    # NN architecture
    parser.add_argument('--char', action='store_true', default=False, help='segment sentences character-wise')

    main(parser.parse_args())
