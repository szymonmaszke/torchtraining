import importlib


def modules_exist(*names):
    for name in names:
        if importlib.util.find_spec(name) is None:
            return False

    return True
