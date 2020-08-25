def is_base(klass):
    return (
        "train" not in klass.__name__.lower() and "eval" not in klass.__name__.lower()
    )
