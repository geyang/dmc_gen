if __name__ == '__main__':
    import sys, jaynes
    from params_proto.neo_hyper import Sweep
    from dmc_gen.train import train
    from dmc_gen.arguments import Args

    # [3, 6, 8]
    ms = [43, 44, 45]

    if 'pydevd' in sys.modules:
        # if False:
        train()
    else:
        with Sweep(Args) as sweep, sweep.product:
            Args.algorithm = ['curl', 'rad', 'sac']

        for i, args in enumerate(sweep[1:]):
            # jaynes.config("supercloud", runner=dict(n_cpu=1, n_gpu=0))
            # jaynes.run(train_fn)
            # print(i, args)
            jaynes.config("vision", launch=dict(ip=f"visiongpu{ms[i]:02d}"))
            jaynes.run(train, args, aug_data_prefix="/afs/csail.mit.edu/u/g/geyang/mit/dmc_gen/custom_vendor/data", )

    jaynes.listen()
