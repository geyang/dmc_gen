
if __name__ == '__main__':
    import jaynes
    from dmc_gen.train import train

    jaynes.config("visiongpu")
    jaynes.run(train, data_aug_prefix = "")
    # jaynes.config("supercloud", runner=dict(n_cpu=1, n_gpu=0))
    # jaynes.run(train_fn)
    jaynes.listen()
