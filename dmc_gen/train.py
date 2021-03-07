import os

import gym
import numpy as np
from tqdm import trange

from dmc_gen import utils
from dmc_gen import wrappers
from dmc_gen.algorithms import make_agent


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


def train(deps=None, **kwargs):
    from ml_logger import logger
    from dmc_gen.config import Args

    Args._update(deps, **kwargs)
    logger.log_params(Args=vars(Args))

    utils.set_seed_everywhere(Args.seed)
    wrappers.VideoWrapper.prefix = wrappers.ColorWrapper.prefix = DMCGEN_DATA

    # Initialize environments
    gym.logger.set_level(40)
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

    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=Args.train_steps,
        batch_size=Args.batch_size
    )

    start_step, episode, episode_reward, episode_step, done = 0, 0, 0, 0, True
    logger.start('train')
    for step in range(start_step, Args.train_steps + 1):
        if done:
            if step > start_step:
                logger.store_metrics({'train/duration': logger.split('train')})
                logger.log_metrics_summary(dict(step=step), default_stats='mean')

            # Evaluate agent periodically
            if step % Args.eval_freq == 0:
                logger.store_metrics(episode=episode)
                with logger.Prefix(metrics="eval/"):
                    evaluate(env, agent, Args.eval_episodes, save_video=f"videos/{step:08d}_train.mp4")
                with logger.Prefix(metrics="eval/test/"):
                    evaluate(test_env, agent, Args.eval_episodes, save_video=f"videos/{step:08d}_test.mp4")
                logger.log_metrics_summary(dict(step=step), default_stats='mean')

            # Save agent periodically
            if step > start_step and step % Args.save_freq == 0:
                logger.save_module(agent, f"models/{step:06d}.pkl", chunk=2_000_000)
                logger.remove(f"models/{step - Args.save_freq:06d}.pkl")
                # torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

            logger.store_metrics(episode_reward=episode_reward, episode=episode + 1, prefix="train/")

            obs = env.reset()
            episode_reward, episode_step, done = 0, 0, False
            episode += 1

        # Sample action for data collection
        if step < Args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # Run training update
        if step >= Args.init_steps:
            num_updates = Args.init_steps if step == Args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, step)

        # Take step
        next_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        episode_reward += reward
        obs = next_obs

        episode_step += 1

    logger.print(f'Completed training for {Args.domain}_{Args.task}/{Args.algo}/{Args.seed}')


if __name__ == '__main__':
    train()
