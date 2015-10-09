from . import rnn

import chainer
import chainer.functions as F

import util
import cPickle as pickle


class Seq2Seq(util.Model):
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
        self.model = chainer.FunctionSet(encoder=encoder.model, decoder=decoder.model)

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
        code_state = self._encode(xs, train=train)   # encode
        _, ys = self.decoder.forward(code_state, ts, train=train)    # decode
        return ys

    def generate(self, xs, **kwargs):
        code_state = self._encode(xs, train=False)
        ids = self.decoder.generate(code_state, **kwargs)
        return ids

    def forward_batch(self, xs_data, ts_data, train=True):
        """Forward computation for one batch and calculate loss."""
        batch_size = xs_data.shape[1]
        volatile = not train

        xs = []
        for x_data in xs_data:
            x = chainer.Variable(x_data, volatile=volatile)
            xs.append(x)
        eos_x = util.id2var(self.decoder.eos_id, batch_size, train=train)
        xs.append(eos_x)  # at least one <EOS> must come at the end

        ts = []
        for t_data in ts_data:
            t = chainer.Variable(t_data, volatile=volatile)
            ts.append(t)

        ys = self.forward(xs, ts, train=train)
        assert len(ys) == len(ts) + 1

        eos_t = util.id2var(self.decoder.eos_id, batch_size, train=train)
        ts.append(eos_t)  # must predict EOS at the end

        loss = 0
        accs = []
        for y, t in zip(ys, ts):
            loss += F.softmax_cross_entropy(y, t)
            accs.append(F.accuracy(y, t).data)
        acc_avg = sum(accs) / len(accs)

        return loss, acc_avg

    def save(self, path):
        with open(path, 'wb') as f:
            p = self.model.parameters[0]
            device = None
            if hasattr(p, 'device'):
                device = p.device
                self.model.to_cpu()

            pickle.dump(self, f)

            if device is not None:
                self.model.to_gpu(device=device)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
