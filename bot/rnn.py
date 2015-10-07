from . import util

import chainer
import chainer.functions as F
from chainer import cuda
import numpy as np


class Rnn(object):
    """Recurrent Neural Network"""

    def __init__(self, emb_dim, vocab_size, layers, suppress_output=False, lstm=False, irnn=False, active=F.relu, eos_id=0):
        """
        Recurrent Neural Network with multiple layers.
        in_dim -> layers[0] -> ... -> layers[-1] -> out_dim (optional)

        :param int emb_dim: dimension of embeddings
        :param int vocab_size: size of vocabulary
        :param layers: dimensions of hidden layers
        :type layers: list of int
        :param bool suppress_output: suppress output
        :param bool lstm: whether to use LSTM
        :param bool irnn: whether to use IRNN
        :param chainer.Function active: activation function between layers of vanilla RNN
        :param int eos_id: ID of <BOS> and <EOS>
        """
        assert not (lstm and irnn)

        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.layers = layers
        self.suppress_output = suppress_output
        self.lstm = lstm
        self.irnn = irnn
        self.active = active
        self.eos_id = eos_id

        # set up NN architecture
        model = chainer.FunctionSet(
            emb=F.EmbedID(vocab_size, emb_dim),
        )
        # add hidden layers
        layer_dims = [emb_dim] + layers
        for i in range(len(layers)):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i+1]
            if lstm:
                linear = F.Linear(in_dim, out_dim*4)
                hidden = F.Linear(out_dim, out_dim*4)
            else:
                linear = F.Linear(in_dim, out_dim)
                hidden = F.Linear(out_dim, out_dim)
                if irnn:
                    # initialize hidden connection with identity matrix
                    hidden.W = np.eye(out_dim)
            setattr(model, 'l{}_x'.format(i+1), linear)
            setattr(model, 'l{}_h'.format(i+1), hidden)
        if not suppress_output:
            # add output layer
            setattr(model, 'l_y', F.Linear(layer_dims[-1], vocab_size))
        self.model = model

    def step(self, state, x):
        h = self.model.emb(x)
        new_state = {}
        for i in range(len(self.layers)):
            layer_num = i + 1
            linear = getattr(self.model, 'l' + str(layer_num) + '_x')
            hidden = getattr(self.model, 'l' + str(layer_num) + '_h')
            last_h = state['h' + str(layer_num)]
            h_in = linear(h) + hidden(last_h)
            if self.lstm:
                last_c = state['c' + str(layer_num)]
                c, h = F.lstm(last_c, h_in)
                new_state['c' + str(layer_num)] = c
            else:
                h = self.active(h_in)
            new_state['h' + str(layer_num)] = h

        if self.suppress_output:
            return new_state
        else:
            y = self.model.l_y(h)
            return new_state, y

    def create_init_state(self, batch_size, train=True, gpu=None):
        """Create initial state (hidden layers) filled with zeros."""
        volatile = not train
        state = {}
        with util.get_device(gpu):
            if gpu is None:
                xp = np
            else:
                xp = cuda.cupy

            for layer_num, l in enumerate(self.layers, 1):
                h_data = xp.zeros((batch_size, l), dtype=np.float32)
                h = chainer.Variable(h_data, volatile=volatile)
                state['h' + str(layer_num)] = h
                if self.lstm:
                    c_data = xp.zeros((batch_size, l), dtype=np.float32)
                    c = chainer.Variable(c_data, volatile=volatile)
                    state['c' + str(layer_num)] = c
        return state

    def forward(self, state, xs, train=True):
        """Forward computation.

        :param state: initial state
        :type state: dict of (string, chainer.Variable)
        :param xs: list of input (EOS is prepended automatically)
        :type xs: list of chainer.Variable
        :return: final state (and unnormalized probabilities)
        """
        batch_size = xs[0].data.shape[0]
        x0 = util.id2var(self.eos_id, batch_size, train)
        ys = []
        for x in [x0] + xs:
            step_out = self.step(state, x)
            if self.suppress_output:
                state = step_out
            else:
                state, y = step_out
                ys.append(y)
        if self.suppress_output:
            return state
        else:
            return state, ys

    def generate(self, state, max_len=50, exp=3):
        """Generate sequence.
        Batch size must be 1, because all samples in batch must have the same length .

        :param state: initial state
        :type state: dict of (string, chainer.Variable)
        :param int max_len: maximum length of output
        :param int exp: exponentiate output distribution by this number
        :rtype: list of chainer.Variable
        """
        assert not self.suppress_output

        # assert that batch size is 1
        for s in state.values():
            assert s.data.shape[0] == 1

        ids = []     # generated sequence
        x = util.id2var(self.eos_id, train=False)
        for i in range(max_len):
            state, y = self.step(state, x)

            # calculate output distribution
            # cast to float64, otherwise probabilities don't sum to 1
            y_data = y.data[0].astype(np.float64)
            # probabilities are adjusted by exponentiating them by ``exp``
            y_data *= exp
            # softmax
            y_data -= np.max(y_data)
            y_data = np.exp(y_data)
            probs = y_data / np.sum(y_data)

            # sample from distribution
            next_id = np.random.choice(probs.size, p=probs)
            ids.append(next_id)

            if next_id == self.eos_id:
                # terminate generateion
                break
            else:
                # create next input
                x = util.id2var(next_id, train=False)

        return ids
