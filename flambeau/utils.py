import numpy as np


def one_hot(x, n_classes):
    embeddings = np.eye(n_classes)
    x = embeddings[x]
    return x


def cross_entropy(input, target, dim=-1, eps=1e-7):
    return -(target * (input + eps).log()).sum(dim)


def softmax(input, dim=-1):
    exp = input.exp()
    return exp / exp.sum(dim, keepdim=True)
