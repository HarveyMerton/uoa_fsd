#!/usr/bin/env python2.7

import gym
import rospy
import rospkg

from airl.train_expert import train_expert_init
from airl.collect_demo import collect_demo_init
from airl.train_imitation import train_immitation_init

from gym import wrappers
import numpy as np

import fsd_env


def unit_test():

    # Create the Gym environment
    env = gym.make('Fsd-v0')
    rospy.loginfo("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('fsd_rl')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = np.ndarray(0)

    # Set training parameters in param server
    # TODO

    # Initialises the algorithm that we are going to use for learning
    # TODO

    # Function testing
    # RESET
    # print("PRE-RESET")
    # i = 0
    # while i < 1000:
    #     state = env.reset()
    #     i = i+1
    #     print(i)
    # print("POST_RESET")

    #
    # #STEP
    # next_action = 0.0
    # next_action.throttle.data = float(-0.5)  # Set throttle to throttle setpoint
    # next_action.steering_angle.data = float(0.0)

    # done = False
    # step_cnt = 1
    # while not done:
    #     print("PRE-STEP %d" % step_cnt)
    #     state, reward, done, _ = env.step(next_action)
    #     #print("Done: " + str(done))
    #     print(state)
    #     print("POST-STEP %d" % step_cnt)
    #     step_cnt = step_cnt + 1

    # make observation
    while True:
        #observation = env.make_observation()
        observation = {
            "cones_yellow": [(0,2), (0,1), (2,3)],
            "cones_blue": [(0,-4), (8,-1.2), (0,-5.5)],
            "steering_angle": 0.5
        }
        print("Observation: ")
        print(observation)

        rospy.sleep(1)

        print("Point: {}".format(env.helper_point_furthest_center(observation)))


if __name__ == '__main__':
    print("Start Testing")

    # Initialise fsd gym environment node
    rospy.init_node('fsd_gym', anonymous=True)

    #unit_test()

    # Train SAC forward reinforcement learning
    # default, num_steps_ip, eval_interval_ip, env_id_ip, cuda_ip, seed_ip
    # ORIG params:
    # train_expert_init(False, 15, 5, 'Fsd-v0', False, 0)  # Short test
    # train_expert_init(False, 10 ** 6, 10 ** 4, 'Fsd-v0', False, 0)  # Long test

    # Collect expert demonstrations
    # default, sac_expert, weight_ip (expert weights for SAC), env_id_ip, buffer_size_ip, std_ip, p_rand_ip, cuda_ip, seed_ip
    # ORIG params: buffer size 10 ** 6
    # collect_demo_init(False, False, None, 'Fsd-v0', 5*10**4, 0.0, 0.0, False, 0) # 7 hr
    # collect_demo_init(False, False, None, 'Fsd-v0', 2000, 0.0, 0.0, False, 0)  # 2 runs of just over 5 laps

    # Train AIRL inverse reinforcement learning
    # default, buffer, rollout_length (2000 for Inv.P and 50000 for HOPPER), num_steps, eval_interval, env_id, algo, cuda, seed
    # ORIG params: size2000
    train_immitation_init(False, 'buffers/Fsd-v0/size15_std0.0_prand0.0.pth', 50000, 100000, 5000, 'Fsd-v0', 'airl', False, 0)

    print("End")




