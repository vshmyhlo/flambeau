import matplotlib.pyplot as plt
import numpy as np

import autograd as flam
import nn
import optim
from autograd import Variable, exp, log


def one_hot(x, n_classes):
    embeddings = np.eye(n_classes)
    x = embeddings[x]
    return x


def cross_entropy(probs, labels):
    return -(labels * log(probs)).sum(-1)


def softmax(logits):
    exped = exp(logits)
    size = (*exped.size()[:-1], 1)
    return exped / exped.sum(-1).view(size)


def compute_loss(logits, labels):
    probs = softmax(logits)
    cross_ent = cross_entropy(probs=probs, labels=labels)
    loss = cross_ent.mean()
    return loss


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 8)
        self.l2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.l1(x)
        x = flam.relu(x)
        x = self.l2(x)
        return x


model = Model()


def shuffle(x, y):
    idx = np.random.permutation(x.shape[0])
    return x[idx], y[idx]


m = 1000
pos_x = np.random.standard_normal((m, 2)) + 1.5
neg_x = np.random.standard_normal((m, 2)) - 1.5
pos_y = np.ones((m,), dtype=np.int32)
neg_y = np.zeros((m,), dtype=np.int32)
x = np.concatenate([pos_x, neg_x], 0)
y = np.concatenate([pos_y, neg_y], 0)
x, y = shuffle(x, y)

# plt.scatter(x[:, 0], x[:, 1], s=2, c=y)
# plt.show()

optimizer = optim.SGD(model.parameters(), lr=0.01)

batch_size = 100
for i in range(x.shape[0] // batch_size):
    optimizer.zero_grad()
    batch_i = np.s_[i * batch_size : (i + 1) * batch_size]
    batch_x, batch_y = x[batch_i], y[batch_i]
    batch_y = one_hot(batch_y, 2)
    batch_x, batch_y = Variable(batch_x), Variable(batch_y)

    logits = model(batch_x)
    loss = compute_loss(logits, labels=batch_y)

    print(loss.data)

    loss.backward(1.0)
    optimizer.step()

y_hat = np.argmax(model(Variable(x)).data, -1)

#
# plt.scatter(x[:, 0], x[:, 1], s=2, c=y_hat)
# plt.show()
