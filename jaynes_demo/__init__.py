def train_fn():
    print('this is working')
    import gym

    env = gym.make('Reacher-v2')
    print(env)


if __name__ == '__main__':
    import jaynes

    jaynes.config("visiongpu")
    jaynes.run(train_fn)
    # jaynes.config("supercloud", runner=dict(n_cpu=1, n_gpu=0))
    # jaynes.run(train_fn)
    jaynes.listen()
