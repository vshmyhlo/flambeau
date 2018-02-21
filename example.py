from autograd import Variable
import numpy as np

# x = Variable(np.ones((3, 3)), requires_grad=True)
x = Variable(np.array(1), requires_grad=True)

y = x * 3

print(y)
print(y.grad_fn)
print()

print(x.grad)
y.backward(1)
print(x.grad)
