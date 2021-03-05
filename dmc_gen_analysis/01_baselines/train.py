if __name__ == '__main__':
    import sys, jaynes
    from params_proto.neo_hyper import Sweep
    from dmc_gen.train import train
    from dmc_gen.config import Args
    from dmc_gen_analysis import instr

    if 'pydevd' in sys.modules:
        train()
    else:
        with Sweep(Args) as sweep, sweep.product:
            with sweep.zip:
                Args.domain = ['walker', 'cartpole', 'ball_in_cup', 'finger']
                Args.task = ['walk', 'swingup', 'catch', 'spin']
            Args.algo = ['curl', 'rad']
            Args.seed = [100, 200, 300]

        for i, args in enumerate(sweep):
            jaynes.config("supercloud", )
            # jaynes.config("vision", launch=dict(ip=f"visiongpu{ms[i]:02d}"))
            thunk = instr(train, args, _job_postfix=f"{Args.domain}-{Args.task}/{Args.algo}/{Args.seed}")
            jaynes.run(thunk)

    jaynes.listen()
