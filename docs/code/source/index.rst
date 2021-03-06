:github_url: https://github.com/szymonmaszke/torchtraining

***********
torchtraining
***********

**torchtraining** is a functional PyTorch neural network training library which
provides high level building blocks and integrations instead of trying to do
everything for you under the hood (most of current approaches, e.g. `pytorch-lightning <https://github.com/PyTorchLightning/pytorch-lightning>`__,
`skorch <https://github.com/skorch-dev/skorch>`__).

Using unified approach (single `**` operator) across `metrics`, `callbacks`
and others all users have to do is inherit from specific object (usually `torchtraining.Operation`)
and implement desired `forward` or use provided building blocks by us.

Tutorials
#########

See `Google Colab <https://colab.research.google.com>`__ tutorials to get
a feel and what's possible with this framework:

  * `Introduction <https://colab.research.google.com/drive/19oI8RlpDT9JZnkW8BbFzrLL1Wse6wD5G?usp=sharing>`__ - quick tour around functionalities with CIFAR100 classification and `tensorboard`
  * `GAN training <https://colab.research.google.com/drive/19oI8RlpDT9JZnkW8BbFzrLL1Wse6wD5G?usp=sharing>`__ - more advanced example and creating you own pipeline components

Modules
#######

Below you can find available modules, so be sure to skim through those to see what's
currently possible.

.. toctree::
   :glob:
   :maxdepth: 1

   packages/*

.. toctree::
   :hidden:

   related

Integrations
############

**Integration are currently WIP, please file any issues you find along the way**

Following are currently available:

  * `comet-ml <https://www.comet.ml/site/>`__ - module `torchtraining.callbacks.comet`, 'Self-hosted and cloud-based meta machine learning platform allowing data
    scientists and teams to track, compare, explain and optimize experiments and models.'
  * `neptune.ai <https://neptune.ai/>`__ - module `torchtraining.callbacks.neptune`, 'The most lightweight experiment management tool that fits any workflow'
  * `tensorboard <https://www.tensorflow.org/tensorboard>`__ - module `torchtraining.callbacks.tensorboard`, 'Visualization and tooling needed for machine learning experimentation'
  * `horovod <https://github.com/horovod/horovod>`__ - module `torchtraining.accelerators.horovod` and `torchtraining.accelerators.Horovod`,
    'Simple distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.'

All of those can be installed via `extras`, see below.

Installation
############

Following installation methods are available:

`pip: <https://pypi.org/project/torchtraining/>`__
================================================

To install latest release:

.. code-block:: shell

  pip install --user torchtraining

To install `nightly` version:

.. code-block:: shell

  pip install --user torchtraining-nightly

`torchtraining` integrations come with `extra`. Simply run:

.. code-block:: shell

  pip install --user torchtraining[neptune, tensorboard]

To install necessary packages for additional `torchtraining.callbacks` modules.
Available extras:

  * `[all]` - install all `extras` in one go
  * `[callbacks]` - third party callbacks: `[neptune, comet, tensorboard]`
  * `[accelerators]` - third party accelerators: `[horovod]`
  * `[horovod]` - `HOROVOD_CUDA_HOME=/path/to/cuda` may be required during
    `pip install`
  * `[neptune]`
  * `[comet]`
  * `[tensorboard]`
