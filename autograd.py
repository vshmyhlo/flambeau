import numpy as np


class Variable(object):
  def __init__(self, data, requires_grad=False, grad_fn=None):
    self.data = np.array(data)
    self.requires_grad = requires_grad
    self.grad_fn = grad_fn
    self.grad = None

  def __neg__(self):
    return neg(self)

  def __mul__(self, rhs):
    return mul(self, rhs)

  def __rmul__(self, lhs):
    return mul(lhs, self)

  def __add__(self, rhs):
    return add(self, rhs)

  def __radd__(self, lhs):
    return add(lhs, self)

  def __truediv__(self, rhs):
    return div(self, rhs)

  def __rtruediv__(self, lhs):
    return div(lhs, self)

  def __pow__(self, rhs):
    return pow(self, rhs)

  def __rpow__(self, lhs):
    return pow(lhs, self)

  def __matmul__(self, rhs):
    return matmul(self, rhs)
  def sum(self, dim=None):
    return sum(self, dim)

  def mean(self, dim=None):
    return mean(self, dim)


  def view(self, shape):
    return view(self, shape)

  def backward(self, gradient):
    gradient = np.array(gradient)
    assert self.data.shape == gradient.shape, "expected gradient of size {}, got {}".format(
        self.size(), gradient.shape)

    if self.grad_fn is None:
      if self.requires_grad:
        if self.grad is None:
          self.grad = np.zeros_like(self.data)
        self.grad += gradient
    else:
      self.grad_fn(gradient)

  def size(self, dim=None):
    if dim is None:
      return self.data.shape
    else:
      return self.data.shape[dim]

  def dim(self):
    return self.data.ndim

  def __str__(self):
    return 'Variable containing:\n{}\n[dtype: {}, size: {}]'.format(
        self.data, self.data.dtype, self.size())


def to_var(x):
  if isinstance(x, Variable):
    return x
  else:
    return Variable(np.array(x))



def broadcast_shape_elementwise(a, b):
  if a == b:
    return a, (), ()
  elif len(a) == 0:
    return b, None, ()
  elif len(b) == 0:
    return a, (), None,
  elif len(a) == len(b):
    size , a_sum_dim, b_sum_dim = (), (), ()

    for i, (a, b) in enumerate( zip(a, b)):
      if a == b:
        size = (*size, a)
      elif a == 1:
        size , a_sum_dim= (*size, b), (*a_sum_dim, i)
      elif b == 1:
        size, b_sum_dim = (*size, a), (*b_sum_dim, i)
      else:
        raise Exception("Can't broadcast {} to {}".format(a, b))

    return size, a_sum_dim, b_sum_dim

  else:
    raise Exception("Can't broadcast {} to {}".format(a, b))


def broadcast_elementwise(lhs, rhs):
  size, a_sum_dim, b_sum_dim = broadcast_shape_elementwise(lhs.size(), rhs.size())

  lhs_data = lhs.data * np.ones(size)
  if lhs.requires_grad:
    lhs_res = Variable(
        lhs_data,
        requires_grad=True,
        grad_fn=BroadcastElementwiseBackward(lhs, size, a_sum_dim))
  else:
    lhs_res = Variable(lhs_data, requires_grad=False)

  rhs_data = rhs.data * np.ones(size)
  if rhs.requires_grad:
    rhs_res = Variable(
        rhs_data,
        requires_grad=True,
        grad_fn=BroadcastElementwiseBackward(rhs, size, b_sum_dim))
  else:
    rhs_res = Variable(rhs_data, requires_grad=False)

  return lhs_res, rhs_res


class BroadcastElementwiseBackward(object):
  def __init__(self, x, size, sum_dim):
    self.x = x
    self.size = size
    self.sum_dim = sum_dim

  def __call__(self, gradient):
    keepdims = False if self.sum_dim is None else True
    self.x.backward(gradient.sum(self.sum_dim, keepdims=keepdims))


def mul(lhs, rhs):
  lhs, rhs = to_var(lhs), to_var(rhs)
  lhs, rhs = broadcast_elementwise(lhs, rhs)
  data = lhs.data * rhs.data

  if lhs.requires_grad or rhs.requires_grad:
    return Variable(data, requires_grad=True, grad_fn=MulBackward(lhs, rhs))
  else:
    return Variable(data, requires_grad=False, grad_fn=None)


class MulBackward(object):
  def __init__(self, lhs, rhs):
    self.lhs = lhs
    self.rhs = rhs

  def __call__(self, gradient):
    self.lhs.backward(gradient * self.rhs.data)
    self.rhs.backward(gradient * self.lhs.data)


def add(lhs, rhs):
  lhs, rhs = to_var(lhs), to_var(rhs)
  lhs, rhs = broadcast_elementwise(lhs, rhs)
  data = lhs.data + rhs.data

  if lhs.requires_grad or rhs.requires_grad:
    return Variable(data, requires_grad=True, grad_fn=AddBackward(lhs, rhs))
  else:
    return Variable(data, requires_grad=False, grad_fn=None)


class AddBackward(object):
  def __init__(self, lhs, rhs):
    self.lhs = lhs
    self.rhs = rhs

  def __call__(self, gradient):
    self.lhs.backward(gradient)
    self.rhs.backward(gradient)


def div(lhs, rhs):
  lhs, rhs = to_var(lhs), to_var(rhs)
  lhs, rhs = broadcast_elementwise(lhs, rhs)
  data = lhs.data / rhs.data

  if lhs.requires_grad or rhs.requires_grad:
    return Variable(data, requires_grad=True, grad_fn=DivBackward(lhs, rhs))
  else:
    return Variable(data, requires_grad=False, grad_fn=None)


class DivBackward(object):
  def __init__(self, lhs, rhs):
    self.lhs = lhs
    self.rhs = rhs

  def __call__(self, gradient):
    dlhs = 1 / self.rhs.data
    drhs = -(self.lhs.data / self.rhs.data**2)
    self.lhs.backward(gradient * dlhs)
    self.rhs.backward(gradient * drhs)


def view(x, shape):
  x = to_var(x)
  data = x.data.reshape(shape)

  if x.requires_grad:
    return Variable(data, requires_grad=True, grad_fn=ViewBackward(x, shape))
  else:
    return Variable(data, requires_grad=False, grad_fn=None)


class ViewBackward(object):
  def __init__(self, x, shape):
    self.x = x
    self.shape = shape

  def __call__(self, gradient):
    assert gradient.shape == self.shape

    self.x.backward(gradient.reshape(self.x.size()))


def sum(x, dim=None):
  x = to_var(x)
  data = x.data.sum(dim)

  if x.requires_grad:
    return Variable(data, requires_grad=True, grad_fn=SumBackward(x, dim))
  else:
    return Variable(data, requires_grad=False, grad_fn=None)


class SumBackward(object):
  def __init__(self, x, dim):
    self.x = x
    self.dim = dim

  def __call__(self, gradient):
    if self.dim is not None:
      gradient = np.expand_dims(gradient, self.dim)
    dx = np.ones_like(self.x.data)
    self.x.backward(gradient * dx)


def mean(x, dim=None):
  x = to_var(x)
  data = x.data.mean(dim)

  if x.requires_grad:
    return Variable(data, requires_grad=True, grad_fn=MeanBackward(x, dim))
  else:
    return Variable(data, requires_grad=False, grad_fn=None)


class MeanBackward(object):
  def __init__(self, x, dim):
    self.x = x
    self.dim = dim

  def __call__(self, gradient):
    if self.dim is not None:
      gradient = np.expand_dims(gradient, self.dim)
    dx = np.ones_like(self.x.data) / np.array(self.x.data.shape)[self.dim].prod()
    self.x.backward(gradient * dx)


def pow(lhs, rhs):
  lhs, rhs = to_var(lhs), to_var(rhs)
  lhs, rhs = broadcast_elementwise(lhs, rhs)
  data = lhs.data**rhs.data

  if lhs.requires_grad or rhs.requires_grad:
    return Variable(data, requires_grad=True, grad_fn=PowBackward(lhs, rhs))
  else:
    return Variable(data, requires_grad=False, grad_fn=None)


class PowBackward(object):
  def __init__(self, lhs, rhs):
    self.lhs = lhs
    self.rhs = rhs

  def __call__(self, gradient):
    dlhs = self.rhs.data * self.lhs.data**(self.rhs.data - 1)
    drhs = self.lhs.data**self.rhs.data * np.log(self.lhs.data)
    self.lhs.backward(gradient * dlhs)
    self.rhs.backward(gradient * drhs)


def max(a, b):
  a, b = to_var(a), to_var(b)
  a, b = broadcast_elementwise(a, b)
  data = np.maximum(a.data, b.data)
  requires_grad = a.requires_grad or b.requires_grad
  grad_fn = MaxBackward(a, b) if requires_grad else None

  return Variable(data, requires_grad=requires_grad, grad_fn=grad_fn)


class MaxBackward(object):
  def __init__(self, a, b):
    self.a = a
    self.b = b

  def __call__(self, gradient):
    a.backward(gradient * (a > b))
    b.backward(gradient * (a <= b))


def relu(x):
  return max(0.0, x)


def matmul(lhs, rhs):
  lhs, rhs = to_var(lhs), to_var(rhs)
  data = lhs.data.dot(rhs.data)

  if lhs.requires_grad or rhs.requires_grad:
    return Variable(data, requires_grad=True, grad_fn=MatmulBackward(lhs, rhs))
  else:
    return Variable(data, requires_grad=False, grad_fn=None)


class MatmulBackward(object):
  def __init__(self, lhs, rhs):
    self.lhs = lhs
    self.rhs = rhs

  def __call__(self, gradient):
    fail
    # print(gradient.shape)
    # print(self.lhs.size())
    # print(self.rhs.size())


def exp(x):
  x = to_var(x)
  data = np.exp(x.data)

  if x.requires_grad:
    return Variable(data, requires_grad=True, grad_fn=ExpBackward(x))
  else:
    return Variable(data, requires_grad=False, grad_fn=None)


class ExpBackward(object):
  def __init__(self, x):
    self.x = x

  def __call__(self, gradient):
    dx = np.exp(self.x.data)
    self.x.backward(gradient * dx)


def neg(x):
  x = to_var(x)
  data = -x.data

  if x.requires_grad:
    return Variable(data, requires_grad=True, grad_fn=NegBackward(x))
  else:
    return Variable(data, requires_grad=False, grad_fn=None)


class NegBackward(object):
  def __init__(self, x):
    self.x = x

  def __call__(self, gradient):
    dx = -1
    self.x.backward(gradient * dx)


def log(x):
  x = to_var(x)
  data = np.log(x.data)

  if x.requires_grad:
    return Variable(data, requires_grad=True, grad_fn=LogBackward(x))
  else:
    return Variable(data, requires_grad=False, grad_fn=None)


class LogBackward(object):
  def __init__(self, x):
    self.x = x

  def __call__(self, gradient):
    dx = 1 / self.x.data
    self.x.backward(gradient * dx)
