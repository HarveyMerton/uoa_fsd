#!/usr/bin/env python

import gym
import rospy
import os
import math
import time
import numpy as np
#import tf

from gym import utils, spaces
from geometry_msgs.msg import Twist, Vector3Stamped, Pose
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, TransformStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty as EmptyTopicMsg
from gym.utils import seeding
from gym.envs.registration import register
from gazebo_connection import GazeboConnection
import tf2_ros

from fsd_common_msgs.msg import ControlCommand
from fssim_common.msg import ResState, State

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int16, Int32

#from vesc_msgs.msg import *


#register the training environment in the gym as an available one
reg = register(
    id='Fsd-v0',
    entry_point='fsd_env:FsdEnv',
    max_episode_steps=1000
    )

# Global constants
THROTTLE_START = float(0.08)  # Starting throttle position (for initial acceleration)
THROTTLE_START_TIME = 3.0  # Time to accelerate for
THROTTLE_SET = float(0.037)  # Set throttle position
NUM_CONES = 3  # Number of cones of each colour stored and used
RANGE = 10  # Range of cameras
STEER_ANG_MIN = -0.4
STEER_ANG_MAX = 0.4

#IDENT_BLUE = 1  # Identifiers for blue and yellow cones
#IDENT_YELLOW = -1

class FsdEnv(gym.Env):

    def __init__(self):
        # Subscribers
        rospy.Subscriber('/control/pure_pursuit/control_command', ControlCommand, self.callback_cmd)  # Control command
        rospy.Subscriber('/fssim/res_state', ResState, self.callback_res_state)  # Vehicle OK?
        rospy.Subscriber('/fssim/stat_lap_count', Int16, self.callback_lap)  # Lap counter
        rospy.Subscriber('/camera/cones', PointCloud2, self.callback_cones)  # Cone locations
        #rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.callback_init_pose)  # Initial pose

        # Publishers
        # Control command
        self.cmd_airl = rospy.Publisher('/control/pure_pursuit/control_command', ControlCommand, queue_size=5)

        # gets training parameters from param server
        self.running_step = rospy.get_param("/running_step")

        # Establish connection with simulator
        self.gazebo = GazeboConnection()
        rospy.set_param('/use_sim_time', 'true')

        # Define observation and action spaces
        # Observation space
        # Note that range is set in sensors_1.yaml
        self.x_low = float(0)
        self.x_high = float(10)
        self.y_low = float(-5)
        self.y_high = float(5)

        cones_x_low = np.full((2*NUM_CONES, 1), self.x_low)
        cones_x_high = np.full((2*NUM_CONES, 1), self.x_high)
        cones_y_low = np.full((2*NUM_CONES, 1), self.y_low)
        cones_y_high = np.full((2*NUM_CONES, 1), self.y_high)
        #cones_col_low = np.full((2*NUM_CONES, 1), min(IDENT_BLUE, IDENT_YELLOW))
        #cones_col_high = np.full((2*NUM_CONES, 1), max(IDENT_BLUE, IDENT_YELLOW))

        cones_low = np.concatenate((cones_x_low, cones_y_low), axis=1)
        cones_high = np.concatenate((cones_x_high, cones_y_high), axis=1)

        # Flatten cones array
        cones_low = cones_low.flatten()
        cones_high = cones_high.flatten()

        self.observation_space = spaces.Box(low=cones_low, high=cones_high, dtype=float)

        # Action space
        self.action_space = spaces.Box(low=float(STEER_ANG_MIN), high=float(STEER_ANG_MAX), shape=(1,), dtype=float)

        # Set tracking instance variables
        self.np_random = self.cnt_step = self.cnt_lap = self.shutdown = 0  # Tracking variables
        self.obs_cmd = self.obs_cones = None  # Last "observed" command and cone positions
        self.observation_prev = None

        self.helper_reset_vars()

    ### CALLBACKS ###
    # Stores the current command sent
    def callback_cmd(self, data_cmd):
        self.obs_cmd = data_cmd

    # Stores current cone locations
    def callback_cones(self, data_pt_cloud):
        self.obs_cones = list(point_cloud2.read_points(data_pt_cloud, skip_nans=True, field_names=("x", "y", "probability_blue", "probability_yellow", "probability_orange", "probability_other")))

    # Stores shutdown state
    def callback_res_state(self, data_res_state):
        self.shutdown = data_res_state.emergency

    # Stores number of laps
    def callback_lap(self, data_cnt_lap):
        self.cnt_lap = data_cnt_lap.data

    # def callback_init_pose(self, data_init_pose):
    #     self.init_pose = data_init_pose

    ### OPEN_AI INTERFACING FUNCTIONS ###
    # A function to initialize the random generator
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Resets the state of the environment and makes initial observation
    def reset(self):
        # 1st: resets the simulation to initial values
        # Kill and restart /automated_res
        os.system("rosnode kill /automated_res")
        rospy.wait_for_message('/fssim/res_state', ResState)

        # 2nd: Unpauses simulation
        self.gazebo.unpauseSim()

        # Would use reset helper here

        # 3rd: takes an observation of the initial condition of the robot
        observation = self.make_observation()

        # 5th: pauses simulation
        self.gazebo.pauseSim()

        # Reset tracking vars and get initial state
        self.helper_reset_vars()
        state = self.helper_state_from_observation(observation)

        return state

    # Performs a single step in the environment
    def step(self, action):
        # Given the action selected by the learning algorithm,
        # we perform the corresponding movement of the robot

        # 1. Set control command (normalised between -1 and 1)
        next_action = ControlCommand()
        next_action.steering_angle.data = action  # Set steering angle to input steering angle

        # 2. Then we send the command to the robot and let it go
        # for running_step seconds
        self.gazebo.unpauseSim()

        seconds_start_step = rospy.get_time()  # Change to get_rostime()??

        # Send command to sim for running time
        while (rospy.get_time() - seconds_start_step) < self.running_step:
            # Send higher throttle command when accelerating at start
            if self.running_step*self.cnt_step < THROTTLE_START_TIME:
                next_action.throttle.data = THROTTLE_START
            else:
                next_action.throttle.data = THROTTLE_SET

            self.cmd_airl.publish(next_action)

        observation = self.make_observation()
        self.gazebo.pauseSim()

        # Get an evaluation based on what happened in the sim
        reward, done = self.process_data(observation, self.observation_prev)
        state = self.helper_state_from_observation(observation)

        # Update variables
        self.cnt_step += 1
        self.observation_prev = observation
        #self.cnt_step_total += 1

        return state, reward, done, {}

    ### PROCESSING FUNCTIONS ###
    # Process simulation step
    def process_data(self, observation, observation_prev):
        # Check if simulation is done (if vehicle has left track)
        done = False
        if self.shutdown == 1:
            done = True

        # Calculate reward
        # reward = self.helper_reward_steps(done)  # Reward for number of steps inside cones
        reward = self.helper_reward_dist_local(observation_prev, done)  # Reward for moving to correct point

        return reward, done

    # Returns relative locations of blue and yellow cones to vehicle and current steering angle
    def make_observation(self):
        cones_blue = list()  # Left side - list of tuples
        cones_yellow = list()  # Right side - list of tuples
        steering_angle = self.obs_cmd.steering_angle.data

        i = 0
        while i < len(self.obs_cones):
            temp = self.obs_cones[i]

            # Assign cones to the correct lists based on the highest probability
            if temp.index(max(temp[2:5])) == 2:  # Cone is blue
                cones_blue.append(temp[0:2])
            elif temp.index(max(temp[2:5])) == 3:  # Cone is yellow
                cones_yellow.append(temp[0:2])
            # elif temp.index(max(temp[2:5])) == 4:  # Cone is orange
            #     # Treat as blue if on left of car, yellow if on right
            i = i + 1

        # Return only NUM_CONES
        cones_yellow = self.helper_closest_n_cones(cones_yellow, False)
        cones_blue = self.helper_closest_n_cones(cones_blue, True)

        observation = {
            "cones_yellow": cones_yellow,
            "cones_blue": cones_blue,
            "steering_angle": steering_angle
        }

        return observation

    ### HELPER FUNCTIONS ###
    # Sets and resets instance variables
    def helper_reset_vars(self):
        self.seed()  # Set random number initialisation seed

        self.cnt_step = 0  # Counter for number of steps
        self.cnt_lap = 0  # Counter for number of laps

        self.obs_cmd = ControlCommand()  # Latest control command
        self.obs_cones = list()  # Cone positions relative to car
        self.shutdown = 0  # Vehicle quit due to emergency
        # self.init_pose = PoseWithCovarianceStamped()

        self.observation_prev = self.make_observation()

    # Returns state of environment from observation (only cone positions)
    def helper_state_from_observation(self, observation):
        cones_blue_array = np.array(observation["cones_blue"])
        cones_yellow_array = np.array(observation["cones_yellow"])
        #cones_identifiers = np.concatenate((np.full((NUM_CONES, 1), IDENT_BLUE), np.full((NUM_CONES, 1), IDENT_YELLOW)), axis=0)
        #cones_identifiers = np.concatenate(np.full((NUM_CONES, 1), np.full(NUM_CONES, 1)), axis=0)

        #observation = np.concatenate((np.concatenate((cones_blue_array, cones_yellow_array), axis=0), cones_identifiers), axis=1)
        observation = np.concatenate((cones_blue_array, cones_yellow_array), axis=0)
        observation = observation.flatten()

        return observation


    ## REWARD FXNS ##
    # Reward for time spent inside cones - more time = higher reward
    def helper_reward_steps(self, done):
        # Reward staying inside cones
        if not done:
            reward = 1
        else:  # If shutdown (from leaving track), large negative reward?
            reward = 0

        # Add reward modification for action that keeps vehicle in track
        # Add no modification if this is the first observation
        # if len(self.observation_prev["cones_yellow"]) == 0:

        # Find distance to left set of cones

        # Find distance to right set of cones

        # Add modification that rewards small steering angles (for smoothness)

        return reward

    # Reward for moving towards centre between furthest cones
    # observation_prev = observation dictionary from previous timestep, out_cones = boolean (yes if car outside cones)
    def helper_reward_dist_local(self, observation_prev, out_cones):
        reward = 0

        if out_cones:  # Penalise for going out of cones
            reward = -10
        else:
            target = self.helper_point_furthest_center(observation_prev)  # Find target point based on previous observation

            if target[0] == 0 and target[1] == 0:  # If invalid target
                reward = 1  # Reward for staying inside cones
            else:
                dist_list = self.helper_find_dist([target])
                dist_to_target = dist_list[0]

                reward = RANGE/dist_to_target  # Reward based on inverse distance to target

        return reward


    ## CONES PROCESSING ##
    # Finds the point (tuple (x,y)) at the centre of the furthest cone pair
    # currently observed. Returns (0,0) if either cone is at 0 or -ve
    def helper_point_furthest_center(self, observation):
        dist_yellow = self.helper_find_dist(observation["cones_yellow"])
        dist_blue = self.helper_find_dist(observation["cones_blue"])

        furthest_yellow = observation["cones_yellow"][dist_yellow.index(max(dist_yellow))]
        furthest_blue = observation["cones_blue"][dist_blue.index(max(dist_blue))]

        # Find center point of furthest cone pair
        if furthest_yellow[0] <= 0 or furthest_blue[0] <= 0:
            midpoint = (0, 0)  # Return (0, 0) if furthest cone of a colour is at a -ve or 0 x
        else:
            midpoint = self.helper_find_midpoint(furthest_yellow, furthest_blue)

        return midpoint

    # Return only the closest NUM_CONES of each colour
    # cone_list is a list of tuples of (x,y) co-ordinates
    # colour_blue = boolean (true if the cone list is blue)
    def helper_closest_n_cones(self, cone_list, colour_blue):
        cones_dist = self.helper_find_dist(cone_list)
        closest_ind = sorted(range(len(cones_dist)), key=lambda sub: cones_dist[sub])[:NUM_CONES]

        # Add N closest cones to list
        cone_list_n = list()
        for cone_ind in closest_ind:
            cone_list_n.append(cone_list[cone_ind])

        # If less than N cones in list, add extra cones
        if len(cone_list_n) < NUM_CONES:
            cone_list_n = self.helper_add_cones_origin(cone_list_n)

        return cone_list_n

    # Find midpoint between two points p1 and p2 ((x,y) tuples)
    def helper_find_midpoint(self, p1, p2):
        x = (p1[0] + p2[0])/2
        y = (p1[1] + p2[1])/2

        midpoint = (x, y)

        return midpoint

    # Finds the distance of a list of points (potentially cones) from the origin
    # point_list is a list of tuples of (x,y) co-ordinates
    def helper_find_dist(self, point_list):
        point_dist = list()

        for point in point_list:
            point_dist.append(math.sqrt(point[0] ** 2 + point[1] ** 2))

        return point_dist

    # Adds cones if less than NUM_CONES is seen in the FOV - adds cones at origin
    def helper_add_cones_origin(self, cone_list_n):
        cone_origin = (0, 0)

        while len(cone_list_n) < NUM_CONES:
            cone_list_n.append(cone_origin)

        return cone_list_n

    # Adds cones if less than NUM_CONES is seen in the FOV - overlaps existing cones
    def helper_add_cones_overlap(self, cone_list_n, colour_blue):
        # Add cone observations at car x and track extremes if none
        if len(cone_list_n) == 0:
            if colour_blue:
                cone_origin = (self.x_low, self.y_low)
            else:
                cone_origin = (self.x_low, self.y_high)

            cone_list_n.append(cone_origin)

        # Add overlapping cones if required
        num_cones_orig = len(cone_list_n)  # Number of cones originally in list
        i = 0  # Index of last cone used to double-up
        while len(cone_list_n) < NUM_CONES:
            cone_list_n.append(cone_list_n[i % num_cones_orig])

            i = i + 1

    # def helper_reset(self):
        # self.gazebo.resetSim() # Creates "backwards in time" errors
        # self.gazebo.resetWorld()
        # self.check_topic_publishers_connection()
        # self.init_desired_pose()
        # self.takeoff_sequence()

        # rospy.logwarn("Setting Initial Pose")
        # reset_broadcast = tf2_ros.StaticTransformBroadcaster()
        # # # # pose_init = PoseWithCovarianceStamped()
        # # # # # pose_init.pose.pose.position, pose_init.pose.pose.orientation = self.track_checks.get_track_init_pos()
        # # # # #self.pub_initialpose.publish(pose_init)
        # t = TransformStamped()
        # t.header.stamp = rospy.get_rostime()
        # t.header.frame_id = "fssim_map"
        # t.child_frame_id = 'fssim/vehicle/base_link'
        # t.transform.translation.z = 0.0
        # t.transform.translation.x = 0.0
        # t.transform.translation.y = 0.0
        # t.transform.rotation.w = 1.0
        # reset_broadcast.sendTransform(t)
        #
        # t.header.frame_id = "map"
        # t.child_frame_id = 'gotthard_base_link'
        # reset_broadcast.sendTransform(t)
        #
        # testState = State()
        # testState.x = 0
        # testState.y = 0
        #
        # self.testPub.publish(testState)


        # print(self.init_pose)


