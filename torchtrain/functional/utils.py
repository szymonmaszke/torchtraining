def docstring(function):
    """Get module and function name and point users to non-functional counter-part.

    Gets functions name, splits on "_" and transforms to UpperCase class name.

    Redirects users to documentation of non-functional counterpart of `function`.
    """
    function.__doc__ = """See `{}.{}` for details.""".format(
        function.__module__,
        "".join([subname.capitalize() for subname in function.__name__.split("_")]),
    )
    return function
