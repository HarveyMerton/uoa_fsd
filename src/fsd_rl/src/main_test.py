#!/usr/bin/env python2.7

import gym
import rospy
import rospkg
import os

from airl.train_expert import train_expert_init
from airl.collect_demo import collect_demo_init
from airl.train_imitation import train_immitation_init
from airl.run_inference import run_inference_init

from gym import wrappers
import numpy as np

import fsd_env


def unit_test():
    
    # Testing physical inference
    weights_path = os.path.join(os.path.dirname(__file__), 'airl/weights/Fsd-v0.pth')
    run_inference_init(False, 'sac', weights_path, 'Fsd-v0', False)


if __name__ == '__main__':
    print("Start Testing")

    # Initialise fsd gym environment node
    rospy.init_node('fsd_gym', anonymous=True)

    #unit_test()

    # Train SAC forward reinforcement learning
    # default, num_steps_ip, eval_interval_ip, env_id_ip, cuda_ip, seed_ip
    # ORIG params:

    #train_expert_init(False, 15, 5, 'Fsd-v0', False, 0)  # Short test
    #train_expert_init(False, 10 ** 6, 10 ** 4, 'Fsd-v0', False, 0)  # Long test

    # Collect expert demonstrations
    # default, sac_expert, weight_ip (expert weights for SAC), env_id_ip, buffer_size_ip, std_ip, p_rand_ip, cuda_ip, seed_ip
    # ORIG params: buffer size 10 ** 6
    #path_abs = os.path.dirname(os.path.abspath(__file__))
    #path_abs = path_abs + '/airl/weights/Fsd-v0.pth'
    #collect_demo_init(False, True, path_abs, 'Fsd-v0', 5*10**5, 0.0, 0.0, False, 0) # 17 hr
    # collect_demo_init(False, False, None, 'Fsd-v0', 2000, 0.0, 0.0, False, 0)  # 2 runs of just over 5 laps

    # Train AIRL inverse reinforcement learning
    # default, buffer, rollout_length (2000 for Inv.P and 50000 for HOPPER), num_steps, eval_interval, env_id, algo, cuda, seed
    # ORIG params: size2000
    # path_abs = os.path.dirname(os.path.abspath(__file__))
    # path_abs = path_abs + '/airl/buffers/Fsd-v0/size700000_std0.0_prand0.0.pth'
    # train_immitation_init(False, path_abs, 50000, 10 ** 6, 10 ** 4, 'Fsd-v0', 'airl', False, 0)

    # Run inference
    #weights_path = os.path.join(os.path.dirname(__file__), 'airl/weights/Fsd-v0-AIRL-13.pth')
    #run_inference_init(True, 'airl', weights_path, 'Fsd-v0', False)
    weights_path = os.path.join(os.path.dirname(__file__), 'airl/weights/Fsd-v0-SAC-18b-190000.pth')
    run_inference_init(True, 'sac', weights_path, 'Fsd-v0', False)

    print("End")
    