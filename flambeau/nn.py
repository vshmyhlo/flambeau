from collections import OrderedDict

import numpy as np

from flambeau import autograd


class Parameter(autograd.Variable):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module(object):
    def __init__(self):
        self.params = OrderedDict()
        self.modules = OrderedDict()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        if not hasattr(self, "params") or not hasattr(self, "modules"):
            raise RuntimeError("Module.__init__() must be called in child constructor")

        for param in self.params:
            yield self.params[param]

        for module in self.modules:
            for param in self.modules[module].parameters():
                yield param

    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        elif name in self.modules:
            return self.modules[name]
        else:
            return super().__getattr__(name)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.params[name] = value
        elif isinstance(value, Module):
            self.modules[name] = value
        else:
            super().__setattr__(name, value)


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = Parameter(np.random.standard_normal((in_features, out_features)) * 0.1)
        self.bias = Parameter(np.zeros((1, out_features)))

    def forward(self, x):
        return x @ self.weight + self.bias
