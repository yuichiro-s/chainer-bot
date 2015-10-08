import numpy as np
import random


def load_corpus(path):
    """Load pairs of ID sequences from file."""
    corpus = []
    with open(path) as f:
        for line in f:
            es = line.split('\t')
            if len(es) == 2:
                xs_str, ts_str = es
                xs = map(int, xs_str.split())
                ts = map(int, ts_str.split())
                corpus.append((xs, ts))
    return corpus


def create_batches(corpus, batch_size, shuffle=True):
    """
    Turn batches into corpus by grouping samples of similar lengths.

    :param corpus: list of pairs of input sequence and output sequence
    :param batch_size: batch size
    :param shuffle: whether to shuffle batches
    :return: list of batches
    """
    sorted_corpus = sorted(corpus, key=lambda (xs, ts): (len(xs), len(ts)))

    xss = []
    tss = []
    x_len = 0
    t_len = 0

    def _mk_batch(xss, tss, x_len, t_len):
        # batch is full
        x_arr = np.empty((batch_size, x_len), dtype=np.int32)
        t_arr = np.empty((batch_size, t_len), dtype=np.int32)
        x_arr.fill(-1)
        t_arr.fill(-1)
        for i, xs_ in enumerate(xss):
            x_arr[i, :len(xs_)] = xs_
        for i, ts_ in enumerate(tss):
            t_arr[i, :len(ts_)] = ts_
        return x_arr, t_arr

    batches = []
    for xs, ts in sorted_corpus:
        x_len = max(x_len, len(xs))
        t_len = max(t_len, len(ts))
        xss.append(xs)
        tss.append(ts)

        if len(xss) == batch_size:
            batch = _mk_batch(xss, tss, x_len, t_len)
            batches.append(batch)

            # next batch
            xss = []
            tss = []
            x_len = 0
            t_len = 0

    if xss:
        batch = _mk_batch(xss, tss, x_len, t_len)
        batches.append(batch)

    # TODO: swap samples in batches to greedily reduce paddings

    # shuffle
    if shuffle:
        random.shuffle(batches)

    return batches

