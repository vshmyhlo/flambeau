import numpy as np
import pytest
import torch

from flambeau import autograd
from flambeau.autograd import Variable


# TODO: rename
def assert_array_eq(a, b):
    assert a is not None
    assert b is not None

    a = np.array(a)
    b = np.array(b)

    assert a.shape == b.shape
    assert np.allclose(a, b)


def test_add():
    a = Variable([1.0, 2.0], requires_grad=True)
    b = Variable([3.0, 4.0], requires_grad=True)
    c = a + b

    assert_array_eq(c.data, [4.0, 6.0])

    c.backward([1.0, 1.0])

    assert_array_eq(a.grad, [1.0, 1.0])
    assert_array_eq(b.grad, [1.0, 1.0])


def test_add_same():
    a = Variable([1.0, 2.0], requires_grad=True)
    c = a + a

    assert_array_eq(c.data, [2.0, 4.0])

    c.backward([1.0, 1.0])

    assert_array_eq(a.grad, [2.0, 2.0])


def test_mul():
    a = Variable([1.0, 2.0], requires_grad=True)
    b = Variable([3.0, 4.0], requires_grad=True)
    c = a * b

    assert_array_eq(c.data, [3.0, 8.0])

    c.backward([1.0, 1.0])

    assert_array_eq(a.grad, [3.0, 4.0])
    assert_array_eq(b.grad, [1.0, 2.0])


def test_div():
    a = Variable([1.0, 2.0], requires_grad=True)
    b = Variable([3.0, 4.0], requires_grad=True)
    c = a / b

    assert_array_eq(c.data, [1.0 / 3.0, 0.5])

    c.backward([1.0, 1.0])

    assert_array_eq(a.grad, [1.0 / 3.0, 0.25])
    assert_array_eq(b.grad, [-1.0 / 9.0, -0.125])


def test_max():
    a = Variable([1.0, 2.0], requires_grad=True)
    b = Variable([2.0, 1.0], requires_grad=True)

    c = autograd.max(a, b)

    assert_array_eq(c.data, [2.0, 2.0])

    c.backward([1.0, 1.0])

    assert_array_eq(a.grad, [0.0, 1.0])
    assert_array_eq(b.grad, [1.0, 0.0])


def test_max_same():
    a = Variable([1.0, 2.0], requires_grad=True)
    b = Variable([1.0, 2.0], requires_grad=True)

    c = autograd.max(a, b)

    assert_array_eq(c.data, [1.0, 2.0])

    c.backward([1.0, 1.0])

    assert_array_eq(a.grad, [0.0, 0.0])
    assert_array_eq(b.grad, [1.0, 1.0])


def test_broadcast_shape_elementwise_scalar_scalar():
    a = ()
    b = ()

    size, a_sum_dim, b_sum_dim = autograd.broadcast_shape_elementwise(a, b)

    assert size == ()
    assert a_sum_dim == ()
    assert b_sum_dim == ()


def test_broadcast_shape_elementwise_scalar_vector():
    a = ()
    b = (2, 3, 4, 5)

    size, a_sum_dim, b_sum_dim = autograd.broadcast_shape_elementwise(a, b)

    assert size == (2, 3, 4, 5)
    assert a_sum_dim == None
    assert b_sum_dim == ()


def test_broadcast_shape_elementwise_vector_vector():
    a = (2, 3, 4, 5)
    b = (2, 1, 4, 1)

    size, a_sum_dim, b_sum_dim = autograd.broadcast_shape_elementwise(a, b)

    assert size == (2, 3, 4, 5)
    assert a_sum_dim == ()
    assert b_sum_dim == (1, 3)


def test_broadcast_shape_elementwise_vector_vector_2():
    a = (2, 3, 4, 1)
    b = (2, 1, 4, 5)

    size, a_sum_dim, b_sum_dim = autograd.broadcast_shape_elementwise(a, b)

    assert size == (2, 3, 4, 5)
    assert a_sum_dim == (3,)
    assert b_sum_dim == (1,)


def test_bradcast_elementwise_scalar_scalar():
    a = Variable(2.0, requires_grad=True)
    b = Variable(3.0, requires_grad=True)
    a_casted, b_casted = autograd.broadcast_elementwise(a, b)

    assert_array_eq(a_casted.data, 2.0)
    assert_array_eq(b_casted.data, 3.0)

    a_casted.backward(1.0)
    b_casted.backward(1.0)

    assert_array_eq(a.grad, 1.0)
    assert_array_eq(b.grad, 1.0)


def test_bradcast_elementwise_scalar_vector():
    a = Variable(2.0, requires_grad=True)
    b = Variable([[3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    a_casted, b_casted = autograd.broadcast_elementwise(a, b)

    assert_array_eq(a_casted.data, [[2.0, 2.0], [2.0, 2.0]])
    assert_array_eq(b_casted.data, [[3.0, 4.0], [5.0, 6.0]])

    a_casted.backward([[1.0, 1.0], [1.0, 1.0]])
    b_casted.backward([[1.0, 1.0], [1.0, 1.0]])

    assert_array_eq(a.grad, 4.0)
    assert_array_eq(b.grad, [[1.0, 1.0], [1.0, 1.0]])


def test_bradcast_elementwise_vector_vector():
    a = Variable([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Variable([[5.0, 6.0]], requires_grad=True)
    a_casted, b_casted = autograd.broadcast_elementwise(a, b)

    assert_array_eq(a_casted.data, [[1.0, 2.0], [3.0, 4.0]])
    assert_array_eq(b_casted.data, [[5.0, 6.0], [5.0, 6.0]])

    a_casted.backward([[1.0, 1.0], [1.0, 1.0]])
    b_casted.backward([[1.0, 1.0], [1.0, 1.0]])

    assert_array_eq(a.grad, [[1.0, 1.0], [1.0, 1.0]])
    assert_array_eq(b.grad, [[2.0, 2.0]])


m = np.random.standard_normal(size=(2, 3, 4))


@pytest.mark.parametrize(
    "data, op",
    [
        # mean
        (m, lambda x: x.mean()),
        (m, lambda x: x.mean(1)),
        (m, lambda x: x.mean(1, keepdim=True)),
        (m, lambda x: x.mean((0, 2))),
        (m, lambda x: x.mean((0, 2), keepdim=True)),
        # # sum
        (m, lambda x: x.sum()),
        (m, lambda x: x.sum(1)),
        (m, lambda x: x.sum(1, keepdim=True)),
        (m, lambda x: x.sum((0, 2))),
        (m, lambda x: x.sum((0, 2), keepdim=True)),
    ],
)
def test_unary(data, op):
    x1 = Variable(data, requires_grad=True)
    y1 = op(x1)
    y1.sum().backward()

    x2 = torch.tensor(data, requires_grad=True)
    y2 = op(x2)
    y2.sum().backward()

    assert_array_eq(y1.data, y2.detach().numpy())
    assert_array_eq(x1.grad, x2.grad.numpy())
