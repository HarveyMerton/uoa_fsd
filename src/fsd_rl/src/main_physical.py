#!/usr/bin/env python2.7

import gym
import rospy
import os

import fsd_env

from airl.run_inference import run_inference_init

if __name__ == '__main__':
    # Initialise fsd gym environment node
    rospy.init_node('fsd_gym', anonymous=True)

    # Run inference
    weights_path = os.path.join(os.path.dirname(__file__), 'airl/weights/Fsd-v0.pth')
    run_inference_init(False, 'sac', weights_path, 'Fsd-v0', False)

    print("End")




