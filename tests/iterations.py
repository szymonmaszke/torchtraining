import torchtraining as tt


def accuracy_pipeline(iteration, writer, name: str):
    return (
        iteration
        ** tt.Select(predictions=1, labels=2)
        ** tt.metrics.classification.multiclass.Accuracy()
        ** tt.accumulators.Mean()
        ** tt.Split(
            tt.callbacks.Log(f"{name} Accuracy"),
            tt.callbacks.tensorboard.Scalar(writer, f"{name}/Accuracy"),
        )
    )


def loss_pipeline(iteration, writer, name: str):
    return (
        iteration
        ** tt.Select(loss=0)
        ** tt.accumulators.Mean()
        ** tt.Split(
            tt.callbacks.Log(f"{name} Loss"),
            tt.callbacks.tensorboard.Scalar(writer, f"{name}/Loss", log="INFO"),
        )
    )


def train(writer, step, module, data, name: str):
    iteration = tt.iterations.Train(step, module, data, log="INFO")
    return accuracy_pipeline(loss_pipeline(iteration, writer, name), writer, name)


def eval(writer, step, module, data, name: str):
    iteration = tt.iterations.Eval(step, module, data, log="INFO")
    return accuracy_pipeline(loss_pipeline(iteration, writer, name), writer, name)
