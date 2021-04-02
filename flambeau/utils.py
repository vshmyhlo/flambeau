import numpy as np


def one_hot(x, n_classes):
    embeddings = np.eye(n_classes)
    x = embeddings[x]
    return x


def cross_entropy(probs, labels, dim=-1, eps=1e-7):
    return -(labels * (probs + eps).log()).sum(dim)


def softmax(logits, dim=-1):
    exp = logits.exp()
    return exp / exp.sum(dim, keepdim=True)
