import warnings

__SAMPLER__ = {}


def register_sampler(name: str):
    def wrapper(cls):
        if __SAMPLER__.get(name, None):
            if __SAMPLER__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __SAMPLER__[name] = cls
        cls.name = name
        return cls

    return wrapper


def get_sampler(name: str, **kwargs):
    if __SAMPLER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __SAMPLER__[name](**kwargs)