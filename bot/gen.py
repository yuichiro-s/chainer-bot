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

    new_kwargs = kwargs.copy()

    # set up IDs to exclude
    exclude_ids = new_kwargs.pop('exclude')
    if exclude_ids is None:
        exclude_ids = []
    no_unk = new_kwargs.pop('no_unk')
    if no_unk:
        exclude_ids.append(vocab.unk_id)
    new_kwargs['exclude_ids'] = exclude_ids
    exclude_ids_first = new_kwargs.pop('exclude_first')
    if exclude_ids_first is None:
        exclude_ids_first = []
    new_kwargs['exclude_ids_first'] = exclude_ids_first

    # generate ID sequence
    w_ids = s2s.generate(xs, **new_kwargs)

    # convert to words
    ws = []
    for w_id in w_ids:
        w = vocab.get_word(w_id)
        ws.append(w)

    # print words
    sen = u''.join(ws)

    return sen
