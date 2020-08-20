"""Core pytorch operations regarding optimization (optimize, schedule) are placed in general tests."""
import torch

import pytest
import torchtrain.pytorch as P


def test_detach():
    detach = P.Detach()
    x = torch.randn(10, requires_grad=True)
    x = detach(x) ** 2
    assert not x.requires_grad


def test_backward():
    backward = P.Backward()
    x = torch.randn(10, requires_grad=True)
    y = x ** 2
    backward(y.sum())
    assert x.grad is not None
