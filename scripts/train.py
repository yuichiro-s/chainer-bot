#!/usr/bin/env python

from bot import util
from bot import corpus
from bot import seq2seq
from bot import rnn

import sys


def _get_status(batch):
    status = []
    x_data, t_data = batch
    src_length, batch_size = x_data.shape
    trg_length = t_data.shape[0]
    status.append(('src', src_length))
    status.append(('trg', trg_length))
    status.append(('batch', batch_size))
    return status


def main(args):
    vocab_size = None

    # create sequence-to-sequence model
    encoder = rnn.Rnn(emb_dim=args.emb, vocab_size=vocab_size, layers=args.hidden, suppress_output=True, lstm=args.lstm)
    decoder = rnn.Rnn(emb_dim=args.emb, vocab_size=vocab_size, layers=args.hidden, suppress_output=False, lstm=args.lstm)
    s2s = seq2seq.Seq2Seq(encoder, decoder)

    # create batches
    print >> sys.stderr, 'Creating batches...'
    batches = corpus.create_batches(corpus, batch_size=args.batch, shuffle=not args.no_shuffle)

    # train
    print >> sys.stderr, 'Training started.'
    optimizer = util.list2optimizer(args.optim)
    util.train(s2s, batches, optimizer, args.model, max_epoch=None, gpu=args.gpu, save_every=args.save_every, get_status=_get_status)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train sequence-to-sequence model.')

    parser.add_argument('data', help='path to training data')
    parser.add_argument('vocab', help='vocabulary file')
    parser.add_argument('model', help='destination of model')

    # NN architecture
    parser.add_argument('--emb', type=int, default=100, help='dimension of embeddings')
    parser.add_argument('--hidden', nargs='+', type=int, default=[300, 300], help='dimensions of hidden layers')
    parser.add_argument('--vocab', type=int, default=10000, help='max vocabulary size (dimension of softmax layer)')
    parser.add_argument('--lstm', action='store_true', default=False, help='use LSTM activations')

    # training options
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--optim', nargs='+', default=['Adam'], help='optimization method supported by chainer (optional arguments can be omitted)')
    parser.add_argument('--no-shuffle', action='store_true', default=False, help='don\'t shuffle training data')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID')
    parser.add_argument('--save-every', type=int, default=1, help='save model every this number of epochs')

    main(parser.parse_args())
