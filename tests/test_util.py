from bot import util
from bot import rnn
from bot import seq2seq

import chainer.optimizers as O
import numpy as np
import pytest
import os


def setup_function(function):
    np.random.seed(0)


def test_train(tmpdir):
    """Test seq2seq can learn to reverse input"""
    encoder = rnn.Rnn(emb_dim=100, vocab_size=5, layers=[100], suppress_output=True, lstm=True)
    decoder = rnn.Rnn(emb_dim=100, vocab_size=5, layers=[100], suppress_output=False, lstm=True)
    s2s = seq2seq.Seq2Seq(encoder, decoder)

    def _create_batch(batch_size=20, lo=0, hi=3, length=10):
        xs_data = np.random.random_integers(lo, hi, (length, batch_size)).astype(np.int32) + 1
        ts_data = xs_data[::-1].copy()  # reversed xs_data
        return xs_data, ts_data

    train_batches = map(lambda _: _create_batch(), range(20))
    test_xs_data, test_ts_data = _create_batch()

    def _get_status(batch):
        status = []
        x_data, t_data = batch
        src_length, batch_size = x_data.shape
        trg_length = t_data.shape[0]
        status.append(('src_len', src_length))
        status.append(('trg_len', trg_length))
        status.append(('batch', batch_size))
        return status

    dest_dir = str(tmpdir.mkdir('model'))
    util.train(s2s, train_batches, O.Adam(), dest_dir, max_epoch=6, log=True, save_every=2, get_status=_get_status)

    # check output model
    assert os.path.exists(os.path.join(dest_dir, 'log'))
    assert os.path.exists(os.path.join(dest_dir, 'epoch1'))
    assert not os.path.exists(os.path.join(dest_dir, 'epoch2'))
    assert os.path.exists(os.path.join(dest_dir, 'epoch3'))
    assert not os.path.exists(os.path.join(dest_dir, 'epoch4'))
    assert os.path.exists(os.path.join(dest_dir, 'epoch5'))
    assert os.path.exists(os.path.join(dest_dir, 'epoch6'))
    assert not os.path.exists(os.path.join(dest_dir, 'epoch7'))

    # check log
    log_line = None
    with open(os.path.join(dest_dir, 'log')) as f:
        log_line = f.readline()
    assert 'src_len' in log_line
    assert 'trg_len' in log_line
    assert 'batch' in log_line

    # measure test accuracy
    test_loss, test_avg = s2s.forward_batch(test_xs_data, test_ts_data, train=False)

    assert test_avg > 0.6


