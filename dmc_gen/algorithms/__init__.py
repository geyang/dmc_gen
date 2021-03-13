from .curl import CURL
from .pad import PAD
from .rad import RAD
from .sac import SAC
from .soda import SODA

algorithm = {
    'sac': SAC,
    'rad': RAD,
    'curl': CURL,
    'pad': PAD,
    'soda': SODA
}


def make_agent(algo, obs_shape, act_shape, args):
    from .. import algorithms

    m = getattr(algorithms, algo.lower())
    return getattr(m, algo.upper())(obs_shape, act_shape, args)
