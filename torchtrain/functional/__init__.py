"""Module providing functions (like metrics, losses) which can be used directly.

It is advised to only use `torchtrain.functional.inputs` and `torchtrain.functional.data`,
rest should be piped via `objects`.

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
        > tt.Select(predictions=1, labels=2)
        > tt.metrics.classification.binary.Accuracy()
        > tt.accumulators.Mean()
        > tt.callbacks.Log("Accuracy")
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
        > tt.Select(accuracy=1)
        > tt.accumulators.Mean()
        > tt.callbacks.Log("Accuracy")
    )

Second approach has the following shortcomings:

    - calculation of metrics is mixed with what your network actually does
    with inputs and what it produces
    - `Step`'s' `forward` function has more limited usage. If user want to calculate
    other metrics they have to change `step` manually instead of simply
    adding another `>` pipe

"""

from . import data, inputs, loss, metrics
