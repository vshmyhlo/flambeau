import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.datasets import make_moons
from tqdm import tqdm

import flambeau.nn as nn
import flambeau.optim as optim
from flambeau.autograd import Variable, relu
from flambeau.utils import cross_entropy, one_hot, softmax

BATCH_SIZE = 256
NUM_STEPS = 1000


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 32)
        self.l2 = nn.Linear(32, 2)

    def forward(self, input):
        input = self.l1(input)
        input = relu(input)
        input = self.l2(input)

        return input


def compute_loss(input, target):
    probs = softmax(input)
    loss = cross_entropy(input=probs, target=target)
    return loss


def main():
    x, y = make_moons(10000, noise=0.15)

    model = Model()
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    lr_decay = np.exp(np.log(0.1) / NUM_STEPS)

    losses = []
    lrs = []
    images = []
    for i in tqdm(range(NUM_STEPS)):
        batch_i = np.random.randint(x.shape[0], size=BATCH_SIZE)
        batch_x, batch_y = x[batch_i], y[batch_i]

        batch_y = one_hot(batch_y, 2)
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)

        logits = model(batch_x)
        loss = compute_loss(input=logits, target=batch_y)
        losses.append(loss.mean().data)
        lrs.append(opt.lr)

        opt.zero_grad()
        loss.mean().backward()
        opt.step()
        opt.lr *= lr_decay

        if i % 10 == 0:
            y_hat = np.argmax(model(Variable(x)).data, -1)
            fig = plt.figure()
            plt.scatter(x[:, 0], x[:, 1], s=2, c=y_hat)
            fig.savefig("./data/fig.png")
            plt.close(fig)
            image = Image.open("./data/fig.png")
            image.load()
            images.append(image)

    images[0].save(
        fp="./data/fig.gif",
        format="GIF",
        append_images=images[1:],
        save_all=True,
        duration=10,
        loop=0,
    )

    fig = plt.figure()
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    fig.savefig("./data/loss.png")
    plt.close(fig)

    fig = plt.figure()
    plt.plot(lrs)
    plt.xlabel("step")
    plt.ylabel("lr")
    plt.yscale("log")
    fig.savefig("./data/lr.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
