def docs(function):
    """Get module and function name and point users to non-functional counter-part.

    Gets functions name, splits on "_" and transforms to UpperCase class name.

    Redirects users to documentation of non-functional counterpart of `function`.
    """
    # removes functional from module
    module = "torchtraining" + function.__module__[21:]
    function.__doc__ = """See `{}.{}` for details.""".format(
        module,
        "".join([subname.capitalize() for subname in function.__name__.split("_")]),
    )
    return function
