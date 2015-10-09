#!/usr/bin/env python

from bot import util


def generate(input_words, s2s, vocab, max_len=50, exp=3):
    # convert to IDs
    xs = []
    for w in input_words:
        w_id = vocab.get_id(w)
        x = util.id2var(w_id, train=False)
        xs.append(x)

    # generate ID sequence
    w_ids = s2s.generate(xs, max_len=max_len, exp=exp)

    # convert to words
    ws = []
    for w_id in w_ids:
        w = vocab.get_word(w_id)
        ws.append(w)

    # print words
    sen = u''.join(ws)

    return sen
