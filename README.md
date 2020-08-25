<img align="left" width="256" height="256" src="https://github.com/szymonmaszke/torchtraining/blob/master/assets/logo.png">

So you want to train neural nets with PyTorch? Here are your options:

- __plain PyTorch__ - a lot of tedious work like writing [metrics](https://github.com/pytorch/pytorch/issues/22439) or `for` loops
- __external frameworks__ - more automated in exchange for less freedom,
[less flexibility](https://github.com/skorch-dev/skorch), [lots of esoteric functions](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/core/datamodule.py) and
[stuff under the hood](https://github.com/fastai/fastai/blob/master/fastai2/optimizer.py)


Enter [__torchtraining__](https://github.com/szymonmaszke/torchtraining/) - we try to get what's best from both worlds while adding:
explicitness, functional approach, easy extensions and freedom to structure your code!

__All of that using single `**` piping operator!__

| Version | Docs | Tests | Coverage | Style | PyPI | Python | PyTorch | Docker | LOC |
|---------|------|-------|----------|-------|------|--------|---------|--------|-----|
| [![Version](https://img.shields.io/static/v1?label=&message=0.0.1&color=377EF0&style=for-the-badge)](https://github.com/szymonmaszke/torchtraining/releases) | [![Documentation](https://img.shields.io/static/v1?label=&message=docs&color=EE4C2C&style=for-the-badge)](https://szymonmaszke.github.io/torchtraining/)  | ![Tests](https://github.com/szymonmaszke/torchtraining/workflows/test/badge.svg) | [![codecov](https://codecov.io/gh/szymonmaszke/torchtrain/branch/master/graph/badge.svg?token=UsOMoUe8Tf)](https://codecov.io/gh/szymonmaszke/torchtrain) | [![codebeat](https://img.shields.io/static/v1?label=&message=CB&color=27A8E0&style=for-the-badge)](https://codebeat.co/projects/github-com-szymonmaszke-torchtraining-master) | [![PyPI](https://img.shields.io/static/v1?label=&message=PyPI&color=377EF0&style=for-the-badge)](https://pypi.org/project/torchtraining/) | [![Python](https://img.shields.io/static/v1?label=&message=>3.6&color=377EF0&style=for-the-badge&logo=python&logoColor=F8C63D)](https://www.python.org/) | [![PyTorch](https://img.shields.io/static/v1?label=&message=1.6.0&color=EE4C2C&style=for-the-badge)](https://pytorch.org/) | [![Docker](https://img.shields.io/static/v1?label=&message=docker&color=309cef&style=for-the-badge)](https://hub.docker.com/r/szymonmaszke/torchtraining) | ![LOC](https://img.shields.io/static/v1?label=&message=3000&color=327E50&style=for-the-badge)

## Tutorials

See tutorials to get a grasp of what's the fuss is all about:

- [__Introduction__](https://colab.research.google.com/drive/19oI8RlpDT9JZnkW8BbFzrLL1Wse6wD5G?usp=sharing) - quick tour around functionalities with CIFAR100 classification
and `tensorboard`.
- [__GAN training__](https://colab.research.google.com/drive/1zdyiQtrAVUkzAlb-cFeb1QzJfIj7C91V?usp=sharing) - more advanced example and creating you own pipeline components.

## Installation

See [documentation](https://szymonmaszke.github.io/torchtraining/)
for full list of extras (e.g. installation with integrations like `horovod`).

To just start you can install via `pip`:

```bash
pip install --user torchtraining

```

## Why `torchtraining`?

There are a lot of training libraries around for a lot of frameworks. Why would
you choose this one?

### `torchtraining` fits you, not the other way around

We think it's impossible to squeeze user's code in an overly strict API.
We __are not__ trying to `fit` everything into a single... `.fit()` method (or `Trainer` god class,
see `40!` [arguments in PyTorch-Lightning trainer](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/trainer/trainer.py#L155)).
This approach has shown time and time again it does not work for more complicated
use cases as one cannot foresee the endless possibilities
of training neural network and data generation user might require.
`torchtraining` gives you building blocks to calculate metrics, log results,
distribute training instead.


### Implement single `forward` instead of 40 methods

Implementing `forward` with `data` argument is __all__ you will __ever__ need (okay, `accumulators` also need `calculate`,
but that's it), we add thin `__call__`.
Compare that to `PyTorch-Lightning`'s `LightningModule` (source code [here](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/core/lightning.py#L51))

- `training_step`
- `training_step_end`
- `training_epoch_end` (repeat all the above for `validation` and `test`)
- `validation_end`, `test_end`
- `configure_sync_batchnorm`
- `configure_ddp`
- `init_ddp_connection`
- `configure_apex`
- `configure_optimizers`
- `optimizer_step`
- `optimizer_zero_grad`
- `tbptt_split_batch` (?)
- `prepare_data`
- `train_dataloader`
- `tng_dataloader`
- `test_dataloader`
- `val_dataloader`

This list could go on (and probably will grow even bigger as time passes).
We believe in functional approach and using only what you need (a lot of decoupled building blocks instead
of gigantic god classes __trying__ to do everything). Once again: we can't foresee
future and __won't__ squash everything into single `class`.

### Explicitness

You are offered building blocks and it's up to you what you want to use.
Still, you are explicit about __everything__ going on in your code, for example:
- when, where and what to log to `tensorboard`
- when and how often to run optimization
- what `neural network(s)` go into what step
- what data you choose to accumulate and how often
- which component of your pipeline should log via `loguru`
- and how to log (e.g. to `stdout` and `file` or maybe over the web?)

See introduction tutorial to see how it's done

### Neural network != training

We don't think your neural network source code should be polluted with training.
We think it's better to have `data` preparation in `data.py` module,
`optimizers` in `optimizers.py` and so on. With `torchtraining` you don't have to
crunch any functionalities into single god `class`.

### Nothing under the hood (almost)

`~3000` lines of code (including `comet-ml`, `neptune` and `horovod` integration)
and short functions/classes allow you to quickly dig
into the source if you find something odd/not working. It's leverages what exists
instead of reinventing the wheel.


### PyTorch first

We don't force you to jump into and from `numpy` as most of the tasks can already be
done in `PyTorch`. We are `pytorch` first.
Unless we have to integrate third party tool... In that case __you don't pay for
this feature if you don't use it!__

### Easy integration with other tools

If we don't provide an integration out of the box, you can request it via `issues`
or make your own `PR`. Any code you want can almost always be integrated via following steps:

- make a new module (say `amazing.py`)
- create new classes inheriting from `torchtraining.Operation`
- implement `forward` for each operation which takes single argument `data`
which can be anything (`Tuple`, `List`, `torch.Tensor`, `str`, whatever really)
- process this data in `forward` and return results
- you have your own operator compatible with `**`!

Other tools integrate components by trying to squash them into their predefined APIs
and/or trying to be smart and guess what the user does (which often fails).
Here's how we do:

__Example of integration of `neptune` image logging:__


```python
import torchtraining as tt

class Image(tt.Operation):
    def __init__(
        self,
        experiment,
        log_name: str,
        image_name: str = None,
        description: str = None,
        timestamp=None,
        experiment=None,
    ):
        super().__init__()
        self.experiment = experiment
        self.log_name = log_name
        self.image_name = image_name
        self.description = description
        self.timestamp = timestamp

    # Always forward some data so it can be reused
    def forward(self, data):
        self.experiment.log_image(
            self.log_name, data, self.image_name, self.description, self.timestamp
        )
        return data
```

## Contributing

This project is currently in it's infancy and __we would love to get some help from you!__
You can find current ideas inside `issues` tagged by `[DISCUSSION]` (see [here](https://github.com/szymonmaszke/torchtraining/issues?q=DISCUSSION)).

- [`accelerators.py` module for distributed training](https://github.com/szymonmaszke/torchtraining/issues/1)
- [`callbacks.py` third party integrations (experiment handlers like `comet-ml` or `neptune`)](https://github.com/szymonmaszke/torchtraining/issues/2)

Also feel free to make your own feature requests and give us your thoughts in `issues`!

__Remember: It's only `0.0.1` version, direction is there but you can be sure
to encounter a lot of bugs along the way at the moment__

### Why `**` as an operator?

Indeed, operators like `|`, `>>` or `>` would be way more intuitive, __but__:
- Those are left associative and would require users to explicitly uses
parentheses around pipes
- `>` cannot be piped as easily
- __Way more__ complicated code on our side to handle `>>` or `|`

Currently `**` seems like a reasonable trade-off, still it may be subject to
change in future.
