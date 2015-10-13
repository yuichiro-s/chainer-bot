#!/usr/bin/env python

from bot import util


def generate(input_words, s2s, vocab, prefix=None, no_unk=False, exclude=None, exclude_first=None, **kwargs):
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

    # set up IDs to exclude
    exclude_ids = [] if exclude is None else exclude
    if no_unk:
        exclude_ids.append(vocab.unk_id)
    exclude_ids_first = [] if exclude_first is None else exclude_first

    # set up prefix
    prefix_ids = [] if prefix is None else map(vocab.get_id, prefix)

    # generate ID sequence
    w_ids = s2s.generate(xs, prefix=prefix_ids, exclude_ids=exclude_ids, exclude_ids_first=exclude_ids_first, **kwargs)

    # convert to words
    ws = []
    for w_id in w_ids:
        w = vocab.get_word(w_id)
        ws.append(w)

    # print words
    sen = u''.join(ws)

    return sen
