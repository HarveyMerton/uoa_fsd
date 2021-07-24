import os
import argparse
import torch

from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import SACExpert
from gail_airl_ppo.utils import collect_demo

from collections import namedtuple

def run(args, sac_expert):
    env = make_env(args.env_id)

    # IF using SAC as the expert
    if sac_expert:
        algo = SACExpert(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if args.cuda else "cpu"),
            path=args.weight
        )
    else:
        algo = False

    buffer = collect_demo(
        env=env,
        sac_expert=sac_expert,
        algo=algo,
        buffer_size=args.buffer_size,
        device=torch.device("cuda" if args.cuda else "cpu"),
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed
    )

    dirname = os.path.dirname(__file__)
    buffer.save(os.path.join(dirname, 'buffers', args.env_id, "size{}_std{}_prand{}.pth".format(args.buffer_size, args.std, args.p_rand)))

def collect_demo_init(default, sac_expert, weight_ip, env_id_ip, buffer_size_ip, std_ip, p_rand_ip, cuda_ip, seed_ip):
    args = namedtuple("args", "weight env_id buffer_size std p_rand cuda seed")

    if default:
        args.weight = 'weights/InvertedPendulum-v2.pth'
        args.env_id = 'InvertedPendulum-v2'
        args.buffer_size = 10**6
        args.std = 0.0
        args.p_rand = 0.0
        args.cuda = False
        args.seed = 0
    else:
        args.weight = weight_ip  # Only applicable if using forward RL algorithm to train
        args.env_id = env_id_ip
        args.buffer_size = buffer_size_ip
        args.std = std_ip
        args.p_rand = p_rand_ip
        args.cuda = cuda_ip
        args.seed = seed_ip

    run(args, sac_expert)


if __name__ == '__main__':
    collect_demo_init(True, True, None, None, None, None, None, None, None)
