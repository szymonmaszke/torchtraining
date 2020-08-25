"""Core pytorch operations regarding optimization (optimize, schedule) are placed in general tests."""
import pytest
import torch
import torchtraining.pytorch as P


def test_backward():
    backward = P.Backward()
    x = torch.randn(10, requires_grad=True)
    y = x ** 2
    backward(y.sum())
    assert x.grad is not None
