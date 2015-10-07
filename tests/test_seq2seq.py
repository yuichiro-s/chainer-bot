from bot.seq2seq import *
from bot import rnn

import pytest
import numpy as np
import chainer


@pytest.yield_fixture
def s2s():
    encoder = rnn.Rnn(emb_dim=3, vocab_size=3, layers=[4, 3], suppress_output=True, lstm=True)
    decoder = rnn.Rnn(emb_dim=3, vocab_size=3, layers=[4, 3], suppress_output=False, lstm=True)
    s2s = Seq2Seq(encoder, decoder)
    yield s2s


@pytest.yield_fixture
def xs():
    xs = []
    for i in range(4):
        x = chainer.Variable(np.ones(2, dtype=np.int32))    # batch = 2, length = 4
        xs.append(x)
    yield xs


@pytest.yield_fixture
def ts():
    ts = []
    for i in range(5):
        t = chainer.Variable(np.ones(2, dtype=np.int32))    # batch = 2, length = 5
        ts.append(t)
    yield ts


def test_forward(s2s, xs, ts):
    ys = s2s.forward(xs, ts)
    assert len(ys) == 5
    assert ys[0].data.shape == (2, 3)   # batch = 2, vocab = 3


def test_generate(s2s, xs):
    ids = s2s.generate(xs)
    assert len(ids) == 50 or ids[-1] == 0

