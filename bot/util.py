import chainer
from chainer import cuda
import chainer.optimizers as O
import numpy as np

import os
import logging
import time
from collections import OrderedDict


class Model(object):

    def forward_batch(self, x_data, t_data, train):
        """Forward computation for one batch.

        :param xs_data: input
        :type xs_data: list of chainer.Variable
        :param ts_data: expected output
        :type ts_data: list of chainer.Variable
        :param bool train: expected output
        :return: loss and accuracy
        """
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(self, path):
        raise NotImplementedError


def id2var(w_id, batch_size=1, train=True):
    return chainer.Variable(np.asarray([w_id], dtype=np.int32).repeat(batch_size), volatile=not train)


def get_device(gpu):
    return cuda.DummyDevice if gpu is None else cuda.Device(gpu)


def list2optimizer(lst):
    """Create chainer optimizer object from list of strings, such as ['SGD', '0.01']"""
    optim_name = lst[0]
    optim_args = map(float, lst[1:])
    optimizer = getattr(O, optim_name)(*optim_args)
    return optimizer


def _status_str(status):
    lst = []
    for k, v in status.items():
        lst.append(k + ':')
        lst.append(str(v))
    return '\t'.join(lst)


def train(model, batches, optimizer, dest_dir, max_epoch=None, gpu=None, log=True, save_every=1):
    """Common training procedure.

    :param Model model: model to train
    :param batches: training data
    :param optimizer: chainer optimizer
    :param dest_dir: destination directory
    :param max_epoch: maximum number of epochs to train (None to train indefinitely)
    :param gpu: ID of GPU (None to use CPU)
    :param log: whether to enable logging
    :param save_every: save every this number of epochs (first epoch and last epoch are always saved)
    """
    # create model directory
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)   # write to stderr
    if log:
        log_path = os.path.join(dest_dir, 'log')
        file_handler = logging.FileHandler(log_path)
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)     # write to file

    # set up optimizer
    optimizer.setup(model.model)

    # training loop
    epoch = 1
    while True:
        if max_epoch is not None and epoch > max_epoch:
            # terminate training
            break

        # train batches
        for i, (x_data, t_data) in enumerate(batches):
            time_start = time.time()

            optimizer.zero_grads()
            loss, acc = model.forward_batch(x_data, t_data, train=True)
            loss.backward()
            optimizer.update()

            time_end = time.time()
            time_delta = time_end - time_start

            # report training status
            status = OrderedDict()
            status['epoch'] = epoch
            status['batch'] = i + 1
            status['loss'] = loss.data      # training loss
            status['acc'] = '{:.2%}'.format(acc)    # training accuracy
            status['time'] = int(time_delta * 1000)     # time in msec
            status['x1'] = x_data.shape[0]
            status['x2'] = x_data.shape[1]
            status['t1'] = t_data.shape[0]
            status['t2'] = t_data.shape[1]
            logger.info(_status_str(status))

        # save model
        if (epoch - 1) % save_every == 0 or epoch == max_epoch:
            dest_path = os.path.join(dest_dir, 'epoch' + str(epoch))
            model.save(dest_path)

        epoch += 1

