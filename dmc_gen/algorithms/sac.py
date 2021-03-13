from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import utils
from ..algorithms import modules as m


class SAC(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, obs_shape, action_shape, args):
        super().__init__()
        self.discount = args.discount
        self.critic_tau = args.critic_tau
        self.encoder_tau = args.encoder_tau
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq

        shared_cnn = m.SharedCNN(obs_shape, args.num_shared_layers, args.num_filters)  # .to(self.device)
        rl_cnn = m.HeadCNN(shared_cnn.out_shape, args.num_head_layers, args.num_filters)  # .to(self.device)
        actor_encoder = m.Encoder(
            shared_cnn,
            rl_cnn,
            m.RLProjection(rl_cnn.out_shape, args.projection_dim)
        )
        critic_encoder = m.Encoder(
            shared_cnn,
            rl_cnn,
            m.RLProjection(rl_cnn.out_shape, args.projection_dim)
        )

        self.actor = m.Actor(actor_encoder, action_shape, args.hidden_dim, args.actor_log_std_min,
                             args.actor_log_std_max)  # .to(self.device)
        self.critic = m.Critic(critic_encoder, action_shape, args.hidden_dim)  # .to(self.device)
        self.critic_target = deepcopy(self.critic)

        self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(args.init_temperature)))  # .to(self.device)
        self.target_entropy = -np.prod(action_shape)

        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
        )
        self.log_alpha_optim = torch.optim.Adam(
            [self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    # def state_dict(self):
    #     return {k: getattr(self, k).state_dict() for k in
    #             "actor critic critic_target log_alpha target_entropy " \
    #             "actor_optim critic_optim log_alpha_optim".split(' ')}

    def train(self, training=True):
        self.is_training = training
        self.actor.train(training)
        self.critic.train(training)

    def eval(self):
        self.train(False)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done):
        from ml_logger import logger
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        logger.store_metrics({'critic/loss': critic_loss})

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

    def update_actor_and_alpha(self, obs, update_alpha=True):
        from ml_logger import logger

        _, pi, log_pi, log_std = self.actor(obs, detach=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        logger.store_metrics(actor_loss=actor_loss)
        # entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if update_alpha:
            self.log_alpha_optim.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            logger.store_metrics({'alpha/loss': alpha_loss, 'alpha/value': self.alpha})

            alpha_loss.backward()
            self.log_alpha_optim.step()

    def soft_update_critic_target(self):
        utils.soft_update_params(
            self.critic.Q1, self.critic_target.Q1, self.critic_tau
        )
        utils.soft_update_params(
            self.critic.Q2, self.critic_target.Q2, self.critic_tau
        )
        utils.soft_update_params(
            self.critic.encoder, self.critic_target.encoder,
            self.encoder_tau
        )

    def update(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        self.update_critic(obs, action, reward, next_obs, not_done)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
