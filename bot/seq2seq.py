from . import rnn

import chainer
import chainer.functions as F

import util


class Seq2Seq(object):
    """Sequence-to-sequence model."""

    def __init__(self, encoder, decoder):
        """
        :param rnn.Rnn encoder: encoder RNN
        :param rnn.Rnn decoder: decoder RNN
        :return:
        """
        assert encoder.suppress_output
        assert not decoder.suppress_output

        self.encoder = encoder
        self.decoder = decoder

    def _encode(self, xs, train=True):
        batch_size = xs[0].data.shape[0]
        init_state = self.encoder.create_init_state(batch_size, train=train)
        code_state = self.encoder.forward(init_state, xs, train=train)
        return code_state

    def forward(self, xs, ts, train=True):
        """
        Forward computation.

        :param xs: sequence to be encoded
        :param ts: correct output sequence
        :return: output sequence of decoder RNN (before softmax)
        """
        code_state = self._encode(xs, train=True)   # encode
        _, ys = self.decoder.forward(code_state, ts, train=True)    # decode
        return ys

    def generate(self, xs, **kwargs):
        code_state = self._encode(xs, train=False)
        ids = self.decoder.generate(code_state, **kwargs)
        return ids

    def train_batch(self, xs_data, ts_data):
        """Train one batch and calculate loss."""
        batch_size = xs_data.shape[1]

        xs = []
        for x_data in xs_data:
            x = chainer.Variable(x_data)
            xs.append(x)

        ts = []
        for t_data in ts_data:
            t = chainer.Variable(t_data)
            ts.append(t)

        ys = self.forward(xs, ts, train=True)
        assert len(ys) == len(ts) + 1

        eos = util.id2var(self.decoder.eos_id, batch_size, train=True)
        ts.append(eos)  # must predict EOS at the end

        loss = 0
        for y, t in zip(ys, ts):
            loss += F.softmax_cross_entropy(y, t)

        return loss
