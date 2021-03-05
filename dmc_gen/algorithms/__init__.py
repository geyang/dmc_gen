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


def make_agent(obs_shape, action_shape, args):
    return algorithm[args.algo](obs_shape, action_shape, args)
