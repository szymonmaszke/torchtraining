"""Module providing functions (like metrics, losses) directly usable.

.. note::

    **IMPORTANT**: This module should be rarely used and non-functional
    counterparts should be preferred.


For example this::

    class TrainStep(tt.steps.Train):
        def forward(self, module, sample):
            images, labels = sample
            # Assume predictions are obtained
            ...
            return loss, predictions, labels


    train_step = TrainStep(criterion, device)


    iteration = (
        tt.iterations.Train(train_step, ...)
        ** tt.Select(predictions=1, labels=2)
        ** tt.metrics.classification.binary.Accuracy()
        ** tt.accumulators.Mean()
        ** tt.callbacks.Log("Accuracy")
    )

Should be preferred instead of this (notice `accuracy` calculation in `step`)::


    class TrainStep(tt.steps.Train):
        def forward(self, module, sample):
            images, labels = sample
            # Assume predictions are obtained
            ...
            accuracy = tt.functional.metrics.classification.binary(predictions, labels)
            return loss, accuracy


    train_step = TrainStep(criterion, device)

    iteration = (
        tt.iterations.Train(train_step, ...)
        ** tt.Select(accuracy=1)
        ** tt.accumulators.Mean()
        ** tt.callbacks.Log("Accuracy")
    )

Second approach has the following shortcomings:

    * calculation of metrics is mixed with what your network actually does with inputs and what it produces
    * Step's' `forward` function has more limited usage. If user wants to calculate other metrics they have to change `step` manually instead of simply adding another `**` pipe


.. note::

    **IMPORTANT**: Only reasonable modules to use (currently) in `functional`
    manner (ha, irony), should be `torchtraining.functional.inputs` and
    `torchtraining.functional.data`.


"""

from . import data, inputs, loss, metrics
