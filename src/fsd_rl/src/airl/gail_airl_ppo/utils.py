from tqdm import tqdm
import numpy as np
import torch
import rospy

from .buffer import Buffer

from fsd_common_msgs.msg import ControlCommand
from std_msgs.msg import Int32


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)

def plot_results_rqt(cnt_episode, reward):
    # Tracking variables and publishers
    pub_episode_cnt = rospy.Publisher('/algo/tracking/episode_cnt', Int32, queue_size=10)
    pub_reward = rospy.Publisher('/algo/tracking/episode_reward_total', Int32, queue_size=10)

    # Publish episode information - for graphing.
    # For loop so rqt-multiplot can pick up
    # r = rospy.Rate(100)
    for _ in range(100):
        cnt_episode_out = Int32()  # Episode/game/generation counter
        cnt_episode_out.data = cnt_episode
        pub_episode_cnt.publish(cnt_episode_out)

        reward_pub = Int32()  # Reward from this generation
        reward_pub.data = reward
        pub_reward.publish(reward_pub)

        # r.sleep()

def collect_demo(env, sac_expert, algo, buffer_size, device, std, p_rand, seed=0):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    total_return = 0.0
    num_episodes = 0

    state = env.reset()
    t = 0
    episode_return = 0.0

    if not sac_expert:
        pp_expert = PPExpert()

    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1

        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            #print("State: {}".format(state))
            if sac_expert:
                action = algo.exploit(state)
            else:
                action = pp_expert.get_expert_action()

            #print("Selected action: {}, type: {}".format(action, type(action)))
            action = add_random_noise(action, std)

        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        buffer.append(state, action, reward, mask, next_state)
        episode_return += reward

        if done:
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0

        state = next_state

    mean_reward = total_return / num_episodes
    print('Mean return of the expert is %f' % mean_reward)
    return buffer


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

