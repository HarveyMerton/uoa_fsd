import numpy as np
import rospy

from fsd_common_msgs.msg import ControlCommand
from std_msgs.msg import Float32

##
# File contains classes for linking main algorithms to other devices
##

# Pure pursuit expert
class PPExpert():
    def __init__(self):
        # Subscribe to pure pursuit expert
        rospy.Subscriber('/control/pure_pursuit/control_command_expert', ControlCommand, self.callback_cmd)

        # Set instance variables for tracking
        self.obs_cmd = ControlCommand()

    ### CALLBACKS ###
    # Stores the current command sent
    def callback_cmd(self, data_cmd):
        self.obs_cmd = data_cmd

    ### FUNCTIONS ###
    def get_expert_action(self):
        return np.array([self.obs_cmd.steering_angle.data])
    
    def get_expert_throttle(self): 
        return np.array([self.obs_cmd.throttle.data])

# Connection to simulator
class SimConnection():
    def __init__(self):
        # Subscribe to pure pursuit expert
        rospy.Subscriber('/control/pure_pursuit/control_command', ControlCommand, self.callback_sa)  # Simulated action

        # Set instance variables for tracking
        self.obs_sa = ControlCommand()

    ### CALLBACKS ###
    # Stores the current command sent
    def callback_sa(self, data_sa):
        self.obs_sa = data_sa

    ### FUNCTIONS ###
    def get_sim_sa(self):
        return np.array([self.obs_sa.steering_angle.data])

# Connection to physical
class PhysicalConnection():
    def __init__(self):
        # Subscribe to actual steering angle and desired steering angle
        rospy.Subscriber('/physical/steering/norm_ang', Float32, self.callback_sa)
        rospy.Subscriber('/control_phys/steering/norm_ang', Float32, self.callback_sa_desired)

        # Set instance variables for tracking
        self.obs_sa = Float32()
        self.des_sa = Float32()

    ### CALLBACKS ###
    # Stores the current steering angle
    def callback_sa(self, data_sa):
        self.obs_sa = data_sa

    # Stores the current desired angle
    def callback_sa_desired(self, data_sa_des):
        self.des_sa = data_sa_des

    ### FUNCTIONS ###
    def get_physical_sa(self):
        return np.array([self.obs_sa.data])

    def get_physical_sa_desired(self):
        return np.array([self.des_sa.data])