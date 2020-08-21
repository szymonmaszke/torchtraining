<img align="left" width="256" height="256" src="https://github.com/szymonmaszke/torchtrain/blob/master/assets/logo.png">

So you want to train neural nets with PyTorch? Here are your options:

- __plain PyTorch__ - a lot of tedious work like writing [metrics](https://github.com/pytorch/pytorch/issues/22439) or `for` loops
- __external frameworks__ - more automated in exchange for less freedom,
[less flexibility](https://github.com/skorch-dev/skorch), [lots of esoteric functions](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/core/datamodule.py) and
[stuff under the hood](https://github.com/fastai/fastai/blob/master/fastai2/optimizer.py)


Enter [__torchtrain__]() - we try to get what's best from both worlds while adding:
[explicitness](), [functional approach](), [extensibility]() and [freedom]() to structure your code!

__All of that using single `>` pipe operator!__

## Tutorials

- [__Introduction__]() - quick tour around functionalities with CIFAR100 classification
and `tensorboard`.
- [__GAN training__]() - more advanced example and creating you own pipeline components.

## Contributing

This project is currently in it's infancy and help would be appreciated.
You can find current ideas inside `issues` tagged by `[DISCUSSION]` (see [here](https://github.com/szymonmaszke/torchtrain/issues?q=DISCUSSION)).

- [`accelerators.py` module for distributed training](https://github.com/szymonmaszke/torchtrain/issues/1)
