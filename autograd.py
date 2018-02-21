import numpy as np


class MulBackward(object):
  def __init__(self, lhs, rhs):
    self.lhs = lhs
    self.rhs = rhs

  def __call__(self, gradient):
    # if is_var(self.lhs):
    self.lhs.backward(gradient * self.rhs.data)
    # if is_var(self.rhs):
    self.rhs.backward(gradient * self.lhs.data)


# class SumBackward(object):

# def __call__(self, gradient):


def to_var(x):
  if isinstance(x, Variable):
    return x
  else:
    return Variable(np.array(x))


class Variable(object):
  def __init__(self, data, requires_grad=False, grad_fn=None):
    self.data = data
    self.requires_grad = requires_grad
    self.grad_fn = grad_fn
    self.grad = None

  def __mul__(self, rhs):
    rhs = to_var(rhs)
    data = self.data * rhs.data

    if self.requires_grad:
      return Variable(data, requires_grad=True, grad_fn=MulBackward(self, rhs))
    else:
      return Variable(data, requires_grad=False, grad_fn=None)

  def __rmul__(self, lhs):
    lhs = to_var(lhs)
    data = lhs.data * self.data

    if self.requires_grad:
      return Variable(data, requires_grad=True, grad_fn=MulBackward(lhs, self))
    else:
      return Variable(data, requires_grad=False, grad_fn=None)

  def sum(self):
    data = self.data.sum()

    if self.requires_grad:
      return Variable(data, requires_grad=True, grad_fn=SumBackward())
    else:
      return Variable(data, requires_grad=False, grad_fn=None)

  def backward(self, gradient):
    if self.grad_fn is None:
      self.grad = gradient
    else:
      self.grad_fn(gradient)

  def size(self, dim=None):
    if dim is None:
      return self.data.shape
    else:
      return self.data.shape[dim]

  def __repr__(self):
    return 'Variable containing:\n{}\n[dtype: {}, size: {}]'.format(
        self.data, self.data.dtype, self.size())
