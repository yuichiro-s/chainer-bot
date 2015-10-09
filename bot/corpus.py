import numpy as np
import random


def load_corpus(path, max_len=None):
    """Load pairs of ID sequences from file."""
    corpus = []
    with open(path) as f:
        for line in f:
            es = line.split('\t')
            if len(es) == 2:
                xs_str, ts_str = es
                xs = map(int, xs_str.split())
                ts = map(int, ts_str.split())
                if max_len is None or (len(xs) < max_len and len(ts) < max_len):
                    corpus.append((xs, ts))
    return corpus


def create_batches(corpus, batch_size, shuffle=True, max_vocab_size=None, unk_id=1, eos_id=0):
    """
    Turn batches into corpus by grouping samples of similar lengths.

    :param corpus: list of pairs of input sequence and output sequence
    :param batch_size: batch size
    :param shuffle: whether to shuffle batches
    :param max_vocab_size: max vocabulary size
    :param unk_id: ID of <UNK>
    :param eos_id: ID of <EOS> (used as padding to make implementation simple)
    :return: list of batches
    """

    # map out-of-vocabulary words to UNK
    for xs, ts in corpus:
        for i in range(len(xs)):
            if xs[i] >= max_vocab_size:
                xs[i] = unk_id
        for i in range(len(ts)):
            if ts[i] >= max_vocab_size:
                ts[i] = unk_id

    sorted_corpus = sorted(corpus, key=lambda (xs, ts): (len(xs), len(ts)))

    xss = []
    tss = []
    x_len = 0
    t_len = 0

    def _mk_batch(xss, tss, x_len, t_len):
        # batch is full
        x_arr = np.empty((x_len, batch_size), dtype=np.int32)
        t_arr = np.empty((t_len, batch_size), dtype=np.int32)
        x_arr.fill(eos_id)
        t_arr.fill(eos_id)
        for i, xs_ in enumerate(xss):
            x_arr[x_len-len(xs_):, i] = xs_
        for i, ts_ in enumerate(tss):
            t_arr[:len(ts_), i] = ts_
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

