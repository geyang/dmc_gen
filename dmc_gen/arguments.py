import os
from params_proto.neo_proto import ParamsProto, Proto, Flag

print('should run once')

class Args(ParamsProto):
    # environment
    aug_data_prefix = os.path.dirname(__file__) + "/../custom_vendor/data"

    domain_name = Proto('walker')
    task_name = Proto('walk')
    frame_stack = Proto(3, dtype=int)
    action_repeat = Proto(4, dtype=int)
    episode_length = Proto(1000, dtype=int)
    eval_mode = Proto('color_hard', dtype=str)

    # agent
    algorithm = Proto('sac', dtype=str)
    train_steps = Proto(500_000, dtype=int)
    discount = Proto(0.99, dtype=float)
    init_steps = Proto(1000, dtype=int)
    batch_size = Proto(128, dtype=int)
    hidden_dim = Proto(1024, dtype=int)

    # actor
    actor_lr = Proto(1e-3, dtype=float)
    actor_beta = Proto(0.9, dtype=float)
    actor_log_std_min = Proto(-10, dtype=float)
    actor_log_std_max = Proto(2, dtype=float)
    actor_update_freq = Proto(2, dtype=int)

    # critic
    critic_lr = Proto(1e-3, dtype=float)
    critic_beta = Proto(0.9, dtype=float)
    critic_tau = Proto(0.01, dtype=float)
    critic_target_update_freq = Proto(2, dtype=int)

    # architecture
    num_shared_layers = Proto(11, dtype=int)
    num_head_layers = Proto(0, dtype=int)
    num_filters = Proto(32, dtype=int)
    projection_dim = Proto(100, dtype=int)
    encoder_tau = Proto(0.05, dtype=float)

    # entropy maximization
    init_temperature = Proto(0.1, dtype=float)
    alpha_lr = Proto(1e-4, dtype=float)
    alpha_beta = Proto(0.5, dtype=float)

    # auxiliary tasks
    aux_lr = Proto(1e-3, dtype=float)
    aux_beta = Proto(0.9, dtype=float)
    aux_update_freq = Proto(2, dtype=int)

    # soda
    soda_batch_size = Proto(256, dtype=int)
    soda_tau = Proto(0.005, dtype=float)

    # eval
    save_freq = Proto(100_000, dtype=int)
    eval_freq = Proto(10_000, dtype=int)
    eval_episodes = Proto(30, dtype=int)

    # misc
    seed = 100
    log_dir = 'logs'
    save_video = Flag(False)

    def __init__(self):
        assert self.algorithm in {'sac', 'rad', 'curl', 'pad', 'soda'}, \
            f'specified algorithm "{self.algorithm}" is not supported'

        assert self.eval_mode in {'train', 'color_easy', 'color_hard', 'video_easy',
                                  'video_hard'}, f'specified mode "{self.eval_mode}" is not supported'
        assert self.seed is not None, 'must provide seed for experiment'
        assert self.log_dir is not None, 'must provide a log directory for experiment'


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--domain_name', default='walker')
    parser.add_argument('--task_name', default='walk')
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--episode_length', default=1000, type=int)
    parser.add_argument('--eval_mode', default='color_hard', type=str)

    # agent
    parser.add_argument('--algorithm', default='sac', type=str)
    parser.add_argument('--train_steps', default='500k', type=str)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)

    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)

    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)

    # architecture
    parser.add_argument('--num_shared_layers', default=11, type=int)
    parser.add_argument('--num_head_layers', default=0, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--projection_dim', default=100, type=int)
    parser.add_argument('--encoder_tau', default=0.05, type=float)

    # entropy maximization
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)

    # auxiliary tasks
    parser.add_argument('--aux_lr', default=1e-3, type=float)
    parser.add_argument('--aux_beta', default=0.9, type=float)
    parser.add_argument('--aux_update_freq', default=2, type=int)

    # soda
    parser.add_argument('--soda_batch_size', default=256, type=int)
    parser.add_argument('--soda_tau', default=0.005, type=float)

    # eval
    parser.add_argument('--save_freq', default='100k', type=str)
    parser.add_argument('--eval_freq', default='10k', type=str)
    parser.add_argument('--eval_episodes', default=30, type=int)

    # misc
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--save_video', default=False, action='store_true')

    args = parser.parse_args()

    assert args.algorithm in {'sac', 'rad', 'curl', 'pad',
                              'soda'}, f'specified algorithm "{args.algorithm}" is not supported'

    assert args.eval_mode in {'train', 'color_easy', 'color_hard', 'video_easy',
                              'video_hard'}, f'specified mode "{args.eval_mode}" is not supported'
    assert args.seed is not None, 'must provide seed for experiment'
    assert args.log_dir is not None, 'must provide a log directory for experiment'

    args.train_steps = int(args.train_steps.replace('k', '000'))
    args.save_freq = int(args.save_freq.replace('k', '000'))
    args.eval_freq = int(args.eval_freq.replace('k', '000'))

    return args
