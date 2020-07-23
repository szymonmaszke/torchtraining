def check(output, target):
    if output.shape != target.shape:
        raise ValueError(
            "Output and target has to be of the same shape! Got {} for output and {} for target".format(
                output.shape, target.shape
            )
        )


def threshold(output, target, threshold):
    return output > threshold, target.bool()
