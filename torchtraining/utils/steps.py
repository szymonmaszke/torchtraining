from . import docs_general


def docstring(header, body):
    def gradient():
        return r"""
        gradient : bool
            Whether to turn gradient on/off (for training/evaluation respectively).
        """

    def device():
        return r"""
        device : torch.device
            Device to which tensors could be casted. Available in `forward` as
            `self.device`
        """

    def wrapper(klass):
        docstring = r"""{}.

        {}

        Parameters
        ----------
        criterion : typing.Callable
            Criterion to use to get loss value. Available in `forward` as `self.criterion`
            attribute.
        """.format(
            header, body
        )

        if docs_general.is_base(klass):
            docstring += gradient()

        klass.__doc__ = docstring + device()
        return klass

    return wrapper
