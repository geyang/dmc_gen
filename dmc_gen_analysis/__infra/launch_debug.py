def torch_cuda():
    import torch
    _ = torch.cuda.is_available()
    print("cuda is available " if _ else "cuda is not available")
    a = torch.FloatTensor([10, 20]).to('cuda')
    print(a @ a.T)


def gym_render():
    import gym
    env = gym.make('Reacher-v2')
    env.reset()
    img = env.render('rgb_array')

    print(img.shape)

def dmc_debug():
    from dm_control import suite
    print(suite)


if __name__ == '__main__':
    import jaynes
    from dmc_gen.train import train

    jaynes.config("supercloud")
    jaynes.run(dmc_debug)
    jaynes.listen()
