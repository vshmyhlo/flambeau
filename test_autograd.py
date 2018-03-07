import numpy as np
import autograd
from autograd import Variable


def assert_eq(a, b):
  np.all(a == b)


def test_add():
  a = Variable([1., 2.])
  b = Variable([3., 4.])
  c = a + b

  assert_eq(c.data, [4., 6.])

  c.backward([1., 1.])

  assert_eq(a.grad, [1., 1.])
  assert_eq(b.grad, [1., 1.])


def test_mul():
  a = Variable([1., 2.])
  b = Variable([3., 4.])
  c = a * b

  assert_eq(c.data, [3., 8.])

  c.backward([1., 1.])

  assert_eq(a.grad, [3., 4.])
  assert_eq(b.grad, [2., 3.])


def test_sum():
  x = Variable([[1., 2.], [3., 4.]], requires_grad=True)
  y = x.sum()

  assert_eq(y.data, 10.)

  y.backward(1.0)

  assert_eq(x.grad, [[1., 1.], [1., 1.]])
