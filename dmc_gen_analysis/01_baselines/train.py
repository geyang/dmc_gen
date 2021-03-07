if __name__ == '__main__':
    import jaynes, sys
    from params_proto.neo_hyper import Sweep
    from dmc_gen.train import train
    from dmc_gen.config import Args
    from dmc_gen_analysis import instr, RUN

    if 'pydevd' in sys.modules and False:
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
            # RUN.job_prefix += "some"
            print(args, Args.seed)
            RUN.job_postfix = f"{Args.domain}-{Args.task}/{Args.algo}/{Args.seed}"
            thunk = instr(train, args, _job_counter=False)
            from ml_logger import logger

            logger.log_text("""
            keys:
            - Args.domain
            - Args.task
            - Args.algo
            - Args.seed
            charts:
            - yKeys: ["episode_reward/mean", "train/episode_reward/mean"]
              xKey: step
            - type: video
              glob: videos/*_train.mp4
            - type: video
              glob: videos/*_test.mp4
            """, ".charts.yml", overwrite=True)
            jaynes.run(thunk)

    jaynes.listen()
