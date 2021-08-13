from tqdm import tqdm
import numpy as np
import torch
import rospy
import time
import os

from datetime import datetime
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

### PLOTTING AND LOGGING ###
# Plot training results on rqt_graph during training
# log_dir = Log directory (None if not to log)
def plot_save_training(log_file, cnt_episode, num_steps, reward):
    # Tracking variables and publishers
    pub_episode_cnt = rospy.Publisher('/algo/tracking/episode_cnt', Int32, queue_size=10)
    pub_reward = rospy.Publisher('/algo/tracking/episode_reward_total', Int32, queue_size=10)

    # Publish episode information - for graphing.
    # For loop so rqt-multiplot can pick up
    for _ in range(100):
        cnt_episode_out = Int32()  # Episode/game/generation counter
        cnt_episode_out.data = cnt_episode
        pub_episode_cnt.publish(cnt_episode_out)

        reward_pub = Int32()  # Reward from this generation
        reward_pub.data = reward
        pub_reward.publish(reward_pub)

    # Save in log if applicable
    if log_file is not None:
        # Write to log file
        log_file.writelines([str(cnt_episode), ' ', str(num_steps), ' ', str(reward), '\n'])

## LOGFILES ##
# General function to make logfile in the correct directory
def log_make(dirname, rel_path, logname, sim, algo, ident):
    env = 'sim' if (sim == True) else 'real'
    time = datetime.now().strftime("%Y%m%d-%H%M")
    filename = r'{}_{}_{}_{}_{}.txt'.format(logname, env, algo, ident, time)
    file_path_name = os.path.join(dirname, rel_path, filename)

    # Make directory
    filepath = os.path.dirname(file_path_name)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    return file_path_name

# Make logfile for training results
def log_make_training(log_dir, maxSteps):
    # Log file setup
    rel_path = 'summary'
    file_path_name = log_make(log_dir, rel_path, 'logTraining', True, os.path.split(os.path.split(log_dir)[0])[1],
                              str(maxSteps))

    return file_path_name

# Open logfile for training results
def log_open_training(log_dir, max_steps):
    if log_dir is not None:
        # Find log name and make directory if required
        log_name = log_make_training(log_dir, max_steps)

        # Open file
        log_file = open(log_name, "a")
        log_file.writelines(['Episode_num', ' ', 'Num_steps', ' ', 'Reward', '\n'])

        return log_file
    else:
        return None

# Collect demo
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
            if sac_expert:
                action = algo.exploit(state)
            else:
                action = pp_expert.get_expert_action()

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