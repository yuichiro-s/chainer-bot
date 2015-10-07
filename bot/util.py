import chainer
from chainer import cuda
import numpy as np


def id2var(w_id, batch_size=1, train=True):
    return chainer.Variable(np.asarray([w_id], dtype=np.int32).repeat(batch_size), volatile=not train)


def get_device(gpu):
    return cuda.DummyDevice if gpu is None else cuda.Device(gpu)


def train(model, batches, optimizer, dest_dir, max_epoch=None, batch_size=128, gpu=None):
    """Training procedure.

    """
    optimizer.setup(model.model)

    # create batches
    batches = None

    epoch = 0
    while True:
        if max_epoch is not None and epoch >= max_epoch:
            break

        for x_data, t_data in batches:
            # train one batch
            optimizer.zero_grads()
            loss = model.train_batch(x_data, t_data)
            loss.backward()
            optimizer.update()

        epoch += 1

