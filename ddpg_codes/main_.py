import math
import gym_donkeycar
import gym
import ddpg_ as ddpg
import os
import numpy as np
import time
# import image_process

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
load = False
load_path = "/ddpg/episode_560"
path_to_app = "/home/tinrafiq/Documents/DonkeySimLinux/donkey_sim.x86_64"
port = 9091
headless = False
conf = { "exe_path" : path_to_app, "port" : port }

aggregate_stats_every = 20
ep_rewads = []
ep_average_rewards = []
average_error = []
last_episode = 0

env = gym.make("donkey-generated-track-v0", conf = conf)
time.sleep(5)
print("Environment is setting up.")

observation, reward, done, info = env.reset()
# process = image_process.process_obsevation
# process_info = image_process.process_info
# observation = process(observation)
# info = process_info(info)
# print(observation.shape)

agent = ddpg.DDPGAgent(alpha = 0.001, beta = 0.002, gamma = 0.99, obs_shape = observation.shape,
                 info_shape = info.shape, batch_size = 8, tau = 0.005, noise = 0.1,
                 mem_size = 10000, min_mem_size = 10, min_action = -1.0, optimizer = None,
                 fc1_dims = 256, fc2_dims = 512, max_action = 1.0, n_actions = 2,
                 fc3_dims = 32, fc4_dims = 128, fc5_dims = 8)

print("Agent is initialized.")

if load:
    agent.load_model(load_path + "/weights")
    last_episode, ep_rewards, av_reward = agent.load_params(load_path + "/")

try:
    for episode in range(last_episode + 1, 20000):
        episode_reward = 0
        step = 0
        error = 0
        observation, reward, done, info = env.reset()
        # observation = process(observation)
        # info = process_info(info)

        while True:
            action = agent.get_action(observation, info)
            act = np.array([action[0], (action[1] + 1.1) / 25.0])
            new_observation, reward, done, new_info = env.step(action=act)

            # new_observation = process(new_observation)
            # new_info = process_info(new_info)

            episode_reward += reward
            step += 1

            agent.remember(state = observation, action = action, reward = reward, done = done,
                           new_state = new_observation, info = info, new_info = new_info)
            agent.learn()

            observation = new_observation
            info = new_info

            if done: break

        ep_rewads.append(episode_reward)
        average_reward = episode_reward/step
        ep_average_rewards.append(average_reward)

        print("Episode: ", episode, 
            "  Episode Total Reward: ", episode_reward, 
            "  Episode Average Reward: ", average_reward, "\n" )

        # if episode % aggregate_stats_every == 0 or episode == 1:

        #     agent.save_model(path = f"ddpg/episode_{episode}/weights")
        #     agent.save_params(episode = episode, episode_reward = ep_rewads,
        #                       av_reward = ep_average_rewards, av_err = None,
        #                       path = f"ddpg/episode_{episode}/")

        print("Epsilon: ", agent.epsilon)

except KeyboardInterrupt:
    print("Simulator Interrupted")
    pass