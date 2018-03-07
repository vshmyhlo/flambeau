import numpy as np
from autograd import matmul, Variable


class Module(object):
  def __init__(self):
    self.params = []
    self.modules = []

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def parameters(self):
    if not hasattr(self, 'params') or not hasattr(self, 'modules'):
      raise RuntimeError(
          "Module.__init__() must be called in child constructor")

    for param in self.params:
      yield param

    for module in self.modules:
      for param in module.parameters():
        yield param


class Linear(Module):
  def __init__(self, in_features, out_features):
    super().__init__()

    self.w = Variable(
        np.random.standard_normal((in_features, out_features)),
        requires_grad=True)
    self.b = Variable(np.zeros((1, out_features)))

  def forward(self, x):
    return x.matmul(self.w) + self.b
