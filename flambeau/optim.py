import numpy as np


class Optimizer(object):
    def __init__(self, params):
        self.params = params

    def zero_grad(self):
        for param in self.params:
            param.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr, momentum=None):
        params = list(params)
        assert len(params) != 0, "optimizer got an empty parameter list"

        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.buffers = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for param, buffer in zip(self.params, self.buffers):
            grad = param.grad

            if self.momentum is not None:
                buffer *= self.momentum
                buffer += grad
                grad = buffer

            param.data -= self.lr * grad
