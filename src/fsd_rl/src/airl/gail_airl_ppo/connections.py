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

# Connection to physical
class PhysicalConnection():
    def __init__(self):
        # Subscribe to pure pursuit expert
        rospy.Subscriber('/physical/steering/norm_ang', Float32, self.callback_sa)

        # Set instance variables for tracking
        self.obs_sa = Float32()

    ### CALLBACKS ###
    # Stores the current command sent
    def callback_sa(self, data_sa):
        self.obs_sa = data_sa

    ### FUNCTIONS ###
    def get_physical_sa(self):
        return np.array([self.obs_sa.data])