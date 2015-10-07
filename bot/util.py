from chainer import cuda


def get_device(gpu):
    return cuda.DummyDevice if gpu is None else cuda.Device(gpu)

