import os

import gym
import numpy as np
from tqdm import trange

from dmc_gen import utils
from dmc_gen import wrappers
from dmc_gen.algorithms import make_agent

gym.logger.set_level(40)


def evaluate(env, agent, num_episodes, save_video=None):
    from ml_logger import logger

    episode_rewards, frames = [], []
    for i in trange(num_episodes, desc="Eval"):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            if save_video:
                frames.append(env.render('rgb_array', width=64, height=64))
            episode_reward += reward

        if save_video:
            logger.save_video(frames, key=save_video)
        logger.store_metrics(episode_reward=episode_reward)
        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)


DMCGEN_DATA = os.environ.get('DMCGEN_DATA', os.environ['HOME'] + "/mit/dmc_gen/custom_vendor/data")


def test_saving(deps=None, **kwargs):
    from ml_logger import logger
    from dmc_gen.config import Args

    Args._update(deps, **kwargs)
    logger.log_params(Args=vars(Args))

    utils.set_seed_everywhere(Args.seed)
    wrappers.VideoWrapper.prefix = wrappers.ColorWrapper.prefix = DMCGEN_DATA

    # Initialize environments
    image_size = 84 if Args.algo == 'sac' else 100
    env = wrappers.make_env(
        domain_name=Args.domain,
        task_name=Args.task,
        seed=Args.seed,
        episode_length=Args.episode_length,
        action_repeat=Args.action_repeat,
        image_size=image_size,
    )
    test_env = wrappers.make_env(
        domain_name=Args.domain,
        task_name=Args.task,
        seed=Args.seed + 42,
        episode_length=Args.episode_length,
        action_repeat=Args.action_repeat,
        image_size=image_size,
        mode=Args.eval_mode
    )

    # Prepare agent
    cropped_obs_shape = (3 * Args.frame_stack, 84, 84)
    agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        args=Args
    ).to(Args.device)

    if Args.load_checkpoint:
        print('Loading from checkpoint:', Args.load_checkpoint)
        logger.load_module(agent, path="models/*.pkl", wd=Args.load_checkpoint, map_location=Args.device)
        step = 10_000
        logger.save_module(agent, f"models/{step:06d}.pkl")

    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=Args.train_steps,
        batch_size=Args.batch_size
    )

    episode, episode_reward, episode_step, done = 0, 0, 0, True
    logger.start('train')
    for step in range(Args.start_step, Args.train_steps + 1):
        if step %
            # Save agent periodically
            if step > Args.start_step and step % Args.save_freq == 0:
                logger.save_module(agent, f"models/{step:06d}.pkl")
                logger.remove(f"models/{step - Args.save_freq:06d}.pkl")
                # torch.save(agent, os.path.join(model_dir, f'{step}.pt'))



if __name__ == '__main__':
    test_saving()
