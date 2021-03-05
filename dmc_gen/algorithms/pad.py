import torch
import torch.nn.functional as F

from ..algorithms import modules as m
from ..algorithms.sac import SAC


class PAD(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.aux_update_freq = args.aux_update_freq
        self.aux_lr = args.aux_lr
        self.aux_beta = args.aux_beta

        shared_cnn = self.critic.encoder.shared_cnn
        aux_cnn = m.HeadCNN(shared_cnn.out_shape, args.num_head_layers, args.num_filters).to(self.device)
        aux_encoder = m.Encoder(
            shared_cnn,
            aux_cnn,
            m.RLProjection(aux_cnn.out_shape, args.projection_dim)
        )
        self.pad_head = m.InverseDynamics(aux_encoder, action_shape, args.hidden_dim).to(self.device)
        self.init_pad_optimizer()
        self.train()

    def train(self, training=True):
        super().train(training)
        if hasattr(self, 'pad_head'):
            self.pad_head.train(training)

    def init_pad_optimizer(self):
        self.pad_optimizer = torch.optim.Adam(
            self.pad_head.parameters(), lr=self.aux_lr, betas=(self.aux_beta, 0.999)
        )

    def update_inverse_dynamics(self, obs, obs_next, action):
        from ml_logger import logger
        assert obs.shape[-1] == 84 and obs_next.shape[-1] == 84

        pred_action = self.pad_head(obs, obs_next)
        pad_loss = F.mse_loss(pred_action, action)

        self.pad_optimizer.zero_grad()
        pad_loss.backward()
        self.pad_optimizer.step()

        logger.store_metrics({'inm/loss': pad_loss})

    def update(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        self.update_critic(obs, action, reward, next_obs, not_done)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

        if step % self.aux_update_freq == 0:
            self.update_inverse_dynamics(obs, next_obs, action)
