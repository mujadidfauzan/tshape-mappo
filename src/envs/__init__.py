"""T-shape transport environments; importing physics does not load GUI or YAML."""


def parallel_env(**kwargs):
    from .tshape_env import TShapeParallelEnv
    return TShapeParallelEnv(**kwargs)


__all__ = ["parallel_env"]
