#!/usr/bin/env python

from bot import util


def generate(input_words, s2s, vocab, **kwargs):
    # convert to IDs
    xs = []
    for w in input_words:
        w_id = vocab.get_id(w)
        x = util.id2var(w_id, train=False)
        xs.append(x)

    if len(xs) == 0:
        # Seq2seq doesn't accept empty input, in which case batch size is unknown
        x = util.id2var(s2s.encoder.eos_id, train=False)
        xs.append(x)

    # generate ID sequence
    w_ids = s2s.generate(xs, **kwargs)

    # convert to words
    ws = []
    for w_id in w_ids:
        w = vocab.get_word(w_id)
        ws.append(w)

    # print words
    sen = u''.join(ws)

    return sen
