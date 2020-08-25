from ... import utils


def docs(header, reduction=None):
    def arguments():
        return r"""
Arguments
---------
data: Tuple[torch.Tensor, torch.Tensor]
    Tuple containing `outputs` from neural network and `targets` (ground truths).
    `outputs` should be of shape :math:`(N, *, C-1)`, where :math:`C` is the number of classes.
    Should contain `logits` (unnormalized probabilities) or `probabilities` after
    `softmax` activation or similar.
    `targets` should be of shape :math:`(N, *)` and contain integers in the range
    :math:`[0, C-1]`

        """

    def wrapper(klass):
        docstring = r"""{}

Works for both logits and probabilities of `output` out of the box.

Parameters
----------
        """.format(
            header
        )
        if reduction is not None:
            docstring += utils.docs_general.reduction_parameter(reduction)

        klass.forward.__doc__ = arguments() + utils.docs_general.returned(reduction)
        klass.__doc__ = docstring
        return klass

    return wrapper
