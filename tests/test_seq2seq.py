from bot.seq2seq import *
from bot import rnn

import pytest
import numpy as np
import chainer


def _mk_xs(batch_size, length, train):
    xs = []
    for i in range(length):
        x = chainer.Variable(np.ones(batch_size, dtype=np.int32), volatile=not train)
        xs.append(x)
    return xs


@pytest.yield_fixture
def s2s():
    encoder = rnn.Rnn(emb_dim=3, vocab_size=3, layers=[4, 3], suppress_output=True, lstm=True)
    decoder = rnn.Rnn(emb_dim=3, vocab_size=3, layers=[4, 3], suppress_output=False, lstm=True)
    s2s = Seq2Seq(encoder, decoder)
    yield s2s


def test_forward(s2s):
    xs = _mk_xs(2, 4, True)
    ts = _mk_xs(2, 5, True)
    ys = s2s.forward(xs, ts)
    assert len(ys) == 6
    assert ys[0].data.shape == (2, 3)   # batch = 2, vocab = 3


def test_generate(s2s):
    xs = _mk_xs(1, 4, False)
    ids = s2s.generate(xs)
    assert len(ids) == 50 or ids[-1] == 0

