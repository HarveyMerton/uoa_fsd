import os
import argparse
from datetime import datetime
import torch

from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import SAC
from gail_airl_ppo.trainer import Trainer

from collections import namedtuple


def run(args):
    env = make_env(args.env_id)
    env_test = make_env(args.env_id)

    time = datetime.now().strftime("%Y%m%d-%H%M")
    dirname = os.path.dirname(__file__)
    log_dir = os.path.join(dirname,
                           'logs', args.env_id, 'sac', "seed{}-{}".format(args.seed, time))
    algo = SAC(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        max_steps=args.num_steps,
        log_dir=log_dir
    )

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()


def train_expert_init(default, num_steps_ip, eval_interval_ip, env_id_ip, cuda_ip, seed_ip):
    args = namedtuple("args", "num_steps eval_interval env_id cuda seed")

    if default:
        args.num_steps = 10 ** 6
        args.eval_interval = 10 ** 4
        args.env_id = 'InvertedPendulum-v2'
        args.cuda = False
        args.seed = 0
    else:
        args.num_steps = num_steps_ip
        args.eval_interval = eval_interval_ip
        args.env_id = env_id_ip
        args.cuda = cuda_ip
        args.seed = seed_ip

    run(args)


if __name__ == '__main__':
    # Run with default values
    train_expert_init(True, None, None, None, None, None)
