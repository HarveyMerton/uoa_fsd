import os
import argparse
from datetime import datetime
import torch

from gail_airl_ppo.env import make_env
from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.algo import ALGOS
from gail_airl_ppo.trainer import Trainer

from collections import namedtuple


def run(args):
    env = make_env(args.env_id)
    env_test = make_env(args.env_id)
    buffer_exp = SerializedBuffer(
        path=args.buffer,
        device=torch.device("cuda" if args.cuda else "cpu")
    )

    algo = ALGOS[args.algo](
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, args.algo, 'seed{}-{}'.format(args.seed, time))

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

def train_immitation_init(default, buffer, rollout_length, num_steps, eval_interval, env_id, algo, cuda, seed):
    args = namedtuple("args", "buffer rollout_length num_steps eval_interval env_id algo cuda seed")

    if default:
        args.buffer = 'buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.0.pth'
        args.rollout_length = 2000
        args.num_steps = 100000
        args.eval_interval = 5000
        args.env_id = 'InvertedPendulum-v2'
        args.algo = 'gail'
        args.cuda = False
        args.seed = 0
    else:
        args.buffer = buffer
        args.rollout_length = rollout_length
        args.num_steps = num_steps
        args.eval_interval = eval_interval
        args.env_id = env_id
        args.algo = algo
        args.cuda = cuda
        args.seed = seed

    run(args)

if __name__ == '__main__':
    # Run with default values
    train_immitation_init(True, None, None, None, None, None, None, None, None)
