import importlib


def module_exists(name):
    return importlib.util.find_spec("horovod.torch") is not None
