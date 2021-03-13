if __name__ == '__main__':
    import jaynes, sys
    from ml_logger import logger
    from params_proto.neo_hyper import Sweep
    from dmc_gen.train import train
    from dmc_gen.config import Args
    from dmc_gen_analysis import instr, RUN

    RUN.restart = True
    RUN.prefix = "{username}/{project}/{file_stem}/{job_name}"
    RUN.job_name = "{job_postfix}"
    ms = [
        # "visiongpu54",
        # "improbable005",
        # "improbable006",
        # "improbable007",
        # "improbable008",
        # "improbable009",
        # "improbable010",
    ]

    if 'pydevd' in sys.modules:
        jaynes.config("local")
    else:
        jaynes.config("supercloud", )

    with Sweep(Args) as sweep, sweep.product:
        with sweep.zip:
            Args.domain = ['walker', 'cartpole', 'ball_in_cup', 'finger']
            Args.task = ['walk', 'swingup', 'catch', 'spin']
        Args.algo = ['pad', 'soda', 'curl', 'rad']
        Args.seed = [100, 300, 400]

    for i, args in enumerate(sweep):
        # jaynes.config("vision", launch=dict(ip=ms[i]))
        RUN.job_postfix = f"{Args.domain}-{Args.task}/{Args.algo}/{Args.seed}"
        thunk = instr(train, args)
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
