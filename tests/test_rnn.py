from bot.rnn import *

import pytest
import numpy as np
import chainer


@pytest.yield_fixture
def x():
    # batch size = 2
    yield chainer.Variable(np.zeros(2, dtype=np.int32))

@pytest.yield_fixture
def xs():
    # batch size = 2
    # embedding dimension = 3
    # length = 4
    xs = []
    for i in range(4):
        x = chainer.Variable(np.zeros(2, dtype=np.int32))
        xs.append(x)
    yield xs


def test_irnn():
    """Hidden-to-hidden connections must be initialized with identity matrices."""
    rnn = Rnn(3, 2, [4, 3], irnn=True)
    assert not rnn.lstm
    assert np.allclose(rnn.model.l1_h.W, np.identity(4))
    assert np.allclose(rnn.model.l2_h.W, np.identity(3))


def test_init_state():
    """Hidden state must be initial with zero vectors."""
    rnn = Rnn(3, 2, [4, 3])
    state = rnn.create_init_state(2)
    assert np.count_nonzero(state['h1'].data) == 0
    assert np.count_nonzero(state['h2'].data) == 0


def test_init_state_lstm():
    """Hidden state of LSTM must be initial with zero vectors."""
    rnn = Rnn(3, 2, [4, 3], lstm=True)
    state = rnn.create_init_state(2)
    assert np.count_nonzero(state['h1'].data) == 0
    assert np.count_nonzero(state['h2'].data) == 0


def test_step(x):
    """Check shapes of outputs of one step."""
    rnn = Rnn(3, 2, [4, 3])
    state = rnn.create_init_state(2)
    new_state, y = rnn.step(state, x)
    assert new_state['h1'].data.shape == (2, 4)
    assert new_state['h2'].data.shape == (2, 3)
    assert 'c1' not in new_state
    assert 'c2' not in new_state
    assert y.data.shape == (2, 2)


def test_step_no_output(x):
    rnn = Rnn(3, 2, [4, 3], suppress_output=True)
    state = rnn.create_init_state(2)
    new_state = rnn.step(state, x)
    assert new_state['h1'].data.shape == (2, 4)
    assert new_state['h2'].data.shape == (2, 3)
    assert 'c1' not in new_state
    assert 'c2' not in new_state


def test_step_lstm(x):
    """Check shapes of outputs of one step."""
    rnn = Rnn(3, 2, [4, 3], lstm=True)
    state = rnn.create_init_state(2)
    new_state, y = rnn.step(state, x)
    assert new_state['h1'].data.shape == (2, 4)
    assert new_state['h2'].data.shape == (2, 3)
    assert new_state['c1'].data.shape == (2, 4)
    assert new_state['c2'].data.shape == (2, 3)
    assert y.data.shape == (2, 2)


def test_step_no_output_lstm(x):
    rnn = Rnn(3, 2, [4, 3], lstm=True, suppress_output=True)
    state = rnn.create_init_state(2)
    new_state = rnn.step(state, x)
    assert new_state['h1'].data.shape == (2, 4)
    assert new_state['h2'].data.shape == (2, 3)
    assert new_state['c1'].data.shape == (2, 4)
    assert new_state['c2'].data.shape == (2, 3)


def test_forward(xs):
    rnn = Rnn(3, 2, [4, 3], lstm=True)
    state = rnn.create_init_state(2)
    last_state, ys = rnn.forward(state, xs)
    assert len(ys) == 4
    assert ys[0].data.shape == (2, 2)   # batch=2, dim=2
    assert last_state['h1'].data.shape == (2, 4)
    assert last_state['h2'].data.shape == (2, 3)
    assert last_state['c1'].data.shape == (2, 4)
    assert last_state['c2'].data.shape == (2, 3)
