if __name__ == '__main__':
    import jaynes, sys
    from ml_logger import logger
    from params_proto.neo_hyper import Sweep
    from dmc_gen.train import train
    from dmc_gen.config import Args
    from dmc_gen_analysis import instr, RUN

    with Sweep(Args) as sweep:

        Args.load_checkpoint = "/geyang/dmc_gen/01_baselines/train/ball_in_cup-catch/rad/100"

        Args.domain = 'ball_in_cup'
        Args.task = 'catch'

        Args.algo = 'rad'
        Args.seed = 100

    for i, args in enumerate(sweep):
        if 'pydevd' in sys.modules:
            jaynes.config('local')
        else:
            # jaynes.config("vision", launch=dict(ip=f"visiongpu{ms[i]:02d}"))
            jaynes.config("supercloud", )

        RUN.job_postfix = f"{Args.domain}-{Args.task}/{Args.algo}/{Args.seed}"

        thunk = instr(train, args, _job_counter=False)

        logger.log_text("""
                    keys:
                    - Args.domain
                    - Args.task
                    - Args.algo
                    - Args.seed
                    charts:
                    - yKey: "episode_reward/mean"
                    xKey: step
                    - yKey: "train/episode_reward/mean"
                    xKey: step
                    - type: video
                    glob: videos/*_train.mp4
                    - type: video
                    glob: videos/*_test.mp4
                    """, ".charts.yml", overwrite=True, dedent=True)
        jaynes.run(thunk)

    jaynes.listen()
