from ... import utils


def docs(header, reduction=None):
    def arguments():
        return r"""
Arguments
---------
data: Tuple[torch.Tensor, torch.Tensor]
    Tuple containing `outputs` from neural network and `targets` (ground truths).
    `outputs` should be of shape :math:`(N, *)` and contain `logits` or `probabilities`.
    `targets` should be of shape :math:`(N, *)` as well and contain `boolean` values
    (or integers from set :math:`{0, 1}`).

        """

    def wrapper(klass):
        docstring = r"""{}

Works for both logits and probabilities of `output`.

If `output` is tensor after `sigmoid` activation
user should change `threshold` to `0.5` for
correct results (default `0.0` corresponds to unnormalized probability a.k.a logits).

Parameters
----------
threshold : float, optional
    Threshold above which prediction is considered to be positive.
    Default: `0.0`
        """.format(
            header
        )
        if reduction is not None:
            docstring += utils.docs_general.reduction_parameter(reduction)

        klass.forward.__doc__ = arguments() + utils.docs_general.returned(reduction)
        klass.__doc__ = docstring
        return klass

    return wrapper
