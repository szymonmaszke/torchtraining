from . import docs_general


def docstring(header, reduction=None):
    def arguments():
        return r"""
        Arguments
        ---------
        data: Tuple[torch.Tensor, torch.Tensor]
            Tuple containing `outputs` from neural network and regression `targets`.
            `outputs` should be of shape :math:`(N, F)`, where :math:`N` is the number of samples,
            :math:`F` is the number of features.
            Should contain `floating` point values.
            `targets` should be in the same shape `outputs` and be of `float` data type as well.

        """

    def wrapper(klass):
        docstring = """{header}."""
        if reduction is not None:
            docstring += docs_general.reduction_parameter(reduction)
        docstring += arguments() + docs_general.returned(reduction)

        klass.__doc__ = docstring
        return klass

    return wrapper
