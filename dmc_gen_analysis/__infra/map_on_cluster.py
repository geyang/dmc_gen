def work_fn():
    print('working')

    # for i in range(100):
    #     print('this is running', i)
    #     sleep(0.5)
    #     sys.stdout.flush()


if __name__ == '__main__':
    import jaynes

    # from dmc_gen.train import train
    for i in range(60):
        jaynes.config("vision", launch=dict(ip=f"visiongpu{i:02d}"))
        jaynes.run(work_fn)

    # for i in range(1):
    #     jaynes.config("vision-gpu")
    #     jaynes.run(work_fn)

    # for i in range(60):
    #     jaynes.config("visiongpu", launch=dict(ip=f"visiongpu{i:02d}"),
    #                   runner=dict(pypath="$HOME/jaynes-debug", work_dir="$HOME/jaynes-debug"), mounts=[], )
    #     jaynes.run(work_fn, aug_data_prefix="/afs/csail.mit.edu/u/g/geyang/mit/dmc_gen/custom_vendor/data")

    # jaynes.config("supercloud", runner=dict(n_cpu=1, n_gpu=0))
    # jaynes.run(train_fn)
    jaynes.listen()

# highly non-rectangular
# custom cuda kernels
# how do fully general tensor product
#
# Does it make it harder to optimize. Higher order
