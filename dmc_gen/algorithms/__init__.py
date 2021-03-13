def make_agent(algo, obs_shape, act_shape, args):
    from importlib import import_module

    m = import_module("dmc_gen.algorithms." + algo.lower())
    return m.__dict__[algo.upper()](obs_shape, act_shape, args)
