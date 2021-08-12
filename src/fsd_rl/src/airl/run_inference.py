import os
import argparse
import torch
import numpy as np
import rospy

from datetime import datetime

from gail_airl_ppo.utils import PPExpert, log_make
from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import SACExpert

from collections import namedtuple

def run(args):
    env = make_env(args.env_id)

    # IF using SAC as the expert
    if args.algo == 'sac':
        algo = SACExpert(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if args.cuda else "cpu"),
            path=args.weight
        )
    else:
        algo = False

    # Create logfile
    file_log = open(log_make_inference(args), "a")
    file_log.writelines(['Step_num', ' ', 'Done_flag', ' ', 'Time_total', ' ', 'Action', ' ', 'Action_expert', '\n'])

    # Set variables
    cnt_step = 0  # Time step counter
    step_size = rospy.get_param("/running_step")

    seed = 0
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if args.sim:
        pp_expert = PPExpert()  # Expert for action comparison
        state = env.reset()
    else:
        # TODO physical initial state
        state = None

    while True:
        cnt_step += 1

        # Perform action
        action = algo.exploit(state)

        if args.sim: # In simulation
            next_state, _, done, _ = env.step(action)
            action_expert = pp_expert.get_expert_action()  # Get expert action for comparison
        else:  # In physical world
            # TODO physical step
            #next_state, _, done, _ = env.step(action) # HERE
            action_expert = None

        # Write to log file
        file_log.writelines([str(cnt_step), ' ', str(done), ' ', str(cnt_step * step_size), ' ', str(action), ' ', str(action_expert), ' ', '\n'])

        # Update state
        state = next_state

        # Check if simulation is complete
        if done:
            break

    file_log.close()
    print("Written to logfile at: {}".format(file_log))

def log_make_inference(args):
    # Log file setup
    dirname = os.path.dirname(__file__)
    rel_path = 'logs/{}/inference/'.format(args.env_id)

    file_path_name = log_make(dirname, rel_path, 'logInference', args.sim, args.algo, os.path.splitext(os.path.basename(args.weight))[0])

    return file_path_name

def run_inference_init(sim_ip, algo_ip, weight_ip, env_id_ip, cuda_ip): #, log_ip):
    args = namedtuple("args", "sim algo weight env_id cuda")

    args.sim = sim_ip
    args.algo = algo_ip
    args.weight = weight_ip
    args.env_id = env_id_ip
    args.cuda = cuda_ip

    run(args)

if __name__ == '__main__':
    weights_path = os.path.join(os.path.dirname(__file__), 'airl/weights/Fsd-v0.pth')
    run_inference_init(True, 'sac', 'weights/Fsd-v0.pth', 'Fsd-v0', False) #, 'logs')
