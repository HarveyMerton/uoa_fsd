import os
import argparse
import torch
import numpy as np
import rospy

from datetime import datetime

from gail_airl_ppo.utils import log_make
from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import SACExpert
from gail_airl_ppo.connections import PPExpert, PhysicalConnection, SimConnection
from std_msgs.msg import Int16

from collections import namedtuple

def run(args):
    env = make_env(args.env_id, sim=args.sim)

    # IF using SAC as the expert
    if args.algo == 'sac':
        algo = SACExpert(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if args.cuda else "cpu"),
            path=args.weight
        )

    # Create logfile
    file_log = open(log_make_inference(args), "a")
    file_log.writelines(['Step_num', ' ', 'Done_flag', ' ', 'Time_total', ' ', 'Action', ' ', 'Action other (expert for sim, desired for physical) ', '\n'])
    #file_log.writelines(['Step_num', ' ', 'Done_flag', ' ', 'Time_total', ' ', 'Action',' ','% Total Cones Passed', ' ', 'Action other (expert for sim otherwise physical) ', '\n'])

    # Set variables
    cnt_step = 0  # Time step counter
    step_size = rospy.get_param("/running_step")

    seed = 0
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    state = env.reset()

    # Initialise expert and
    # subscribe to actually executed actions (accounts for env limits)
    if args.sim:
        pp_expert = PPExpert()  # Expert for action comparison
        sim_system = SimConnection()  # Simulated action
    else:
        phys_system = PhysicalConnection()  # Physical action


    while True:
        cnt_step += 1

        # Perform action
        action = algo.exploit(state)  # Select action

        if args.sim:  # In simulation
            next_state, _, done, _ = env.step(action)  # Take step in simulated environment
            action_actual = sim_system.get_sim_sa()  # Get actually executed sa
            action_other = pp_expert.get_expert_action()  # Get expert action for comparison
            #percentage_cones = env.helper_pass_cone_count() # Get percentage of passed cones
        else:  # In physical world
            next_state, _, done, _ = env.step(action)  # Take step in real environment
            action_actual = phys_system.get_physical_sa()  # Get current physical steering angle
            action_other = 0 #phys_system.get_physical_sa_desired()  # Get commanded physical steering angle for comparison (CAUSES CODE TO PAUSE)

        # Write to log file
        file_log.writelines([str(cnt_step), ' ', str(done), ' ', str(cnt_step * step_size), ' ', str(action_actual), ' ', str(action_other), ' ', '\n'])
        #file_log.writelines([str(cnt_step), ' ', str(done), ' ', str(cnt_step * step_size), ' ', str(action_actual), ' ',str(percentage_cones),' ', str(action_other), ' ', '\n'])

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

def run_inference_init(sim_ip, algo_ip, weight_ip, env_id_ip, cuda_ip):
    args = namedtuple("args", "sim algo weight env_id cuda")

    args.sim = sim_ip
    args.algo = algo_ip
    args.weight = weight_ip
    args.env_id = env_id_ip
    args.cuda = cuda_ip

    run(args)

if __name__ == '__main__':
    weights_path = os.path.join(os.path.dirname(__file__), 'airl/weights/Fsd-v0.pth')
    run_inference_init(True, 'sac', 'weights/Fsd-v0.pth', 'Fsd-v0', False)
