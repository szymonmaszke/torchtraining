def returned(reduction):
    if reduction is not None:
        return r"""
Returns
-------
torch.Tensor
    If `reduction` is left as default {} is taken and single value returned.
    Otherwise whatever `reduction` returns.

            """.format(
            reduction
        )
    return r"""
Returns
-------
torch.Tensor
    Scalar `tensor`

        """


def reduction_parameter(reduction):
    if reduction == "mean":
        return r"""
reduction: Callable(torch.Tensor) -> Any, optional
    One argument callable getting `torch.Tensor` and returning argument
    after specified reduction.
    Default: `torch.mean` (mean across batch, user can use `torchtraining.savers.Mean`
    to get mean across iterations/epochs).
"""

    if reduction == "sum":
        return r"""
reduction: Callable, optional
    One argument callable getting `torch.Tensor` and returning `torch.Tensor`.
    Default: `torch.sum` (sum of all elements, user can use `torchtraining.savers.Sum`
    to get sum across iterations/epochs).
"""

    raise ValueError(
        """reduction argument can be one of ["sum", "mean"], got {}""".format(reduction)
    )
