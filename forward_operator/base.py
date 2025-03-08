from abc import ABC, abstractmethod
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import warnings

__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            if __OPERATOR__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __OPERATOR__[name] = cls
        cls.name = name
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class Operator(ABC):
    def __init__(self, sigma=0.05):
        self.sigma = sigma

    @abstractmethod
    def __call__(self, x):
        pass

    def measure(self, x):
        y0 = self(x)
        return y0 + self.sigma * torch.randn_like(y0)

    def loss(self, x, y):
        return ((self(x) - y) ** 2).flatten(1).sum(-1)

    def gradient(self, x, y, return_loss=False):
        x_tmp = x.clone().detach().requires_grad_(True)
        loss = self.loss(x_tmp, y).sum()
        x_grad = torch.autograd.grad(loss, x_tmp)[0]
        if return_loss:
            return x_grad, loss
        return x_grad

    def log_likelihood(self, x, y):
        return -self.loss(x, y) / 2 / self.sigma ** 2

    def likelihood(self, x, y):
        return torch.exp(self.log_likelihood(x, y))

