import math
import gym_donkeycar
import gym
import ddqn
import os
import numpy as np
import time

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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
#* observation = #80*80*1
observation = np.stack((observation, observation, observation), axis = 2) #80*80*3*1
observation = observation.reshape((observation.shape[0], 
                    observation.shape[1], observation.shape[2])) # 80*80*3
# info = process_info(info)

agent = ddqn.DDQNAgent(alpha = 0.001, gamma = 0.999, epsilon = 1.0, 
                       obs_shape = observation.shape, info_shape = info.shape, 
                       batch_size = 64, epsilon_dec = 0.999, epsilon_end = 0.05,
                       mem_size = 10000, min_mem_size = 100, replace_target = 1000,
                       fc1_dims = 512, fc2_dims = 512, fc3_dims = 256, fc4_dims = 256,
                       fc5_dims = 512)

print("Agent is initialized.")

try:
    for episode in range(last_episode + 1, 20000):
        episode_reward = 0
        step = 0
        error = 0
        observation, reward, done, info = env.reset()
        #* observation = #80*80*1
        observation = np.stack((observation, observation, observation), axis = 2) #80*80*3*1
        observation = observation.reshape((observation.shape[0], 
                    observation.shape[1], observation.shape[2])) #! 80*80*3

        while True:
            action, action_index = agent.get_action(observation, info)
            new_observation, reward, done, new_info = env.step(action=action)
            #* new_observation = #80*80*1
            new_observation = np.append(new_observation, observation[:,:,:2], axis = 2) #!80*80*3

            episode_reward += reward
            step += 1

            agent.remember(state=observation, action=action_index, done=done, info=info,
                           reward=reward, new_state=new_observation, new_info=new_info)
            agent.train()

            observation = new_observation
            info = new_info

            if done: break

        ep_rewads.append(episode_reward)
        average_reward = episode_reward/step
        ep_average_rewards.append(average_reward)

        print("Episode: ", episode, 
            "  Episode Total Reward: ", episode_reward, 
            "  Episode Average Reward: ", average_reward, "\n" )
        
        print(agent.epsilon)
        # if episode % aggregate_stats_every == 0 or episode == 1:

            # agent.save_weights(path = f"ddqn/episode_{episode}/weights/")
            # agent.save_params(episode = episode, episode_reward = ep_rewads,
            #                   av_reward = ep_average_rewards, av_err = None,
            #                   path = f"ddqn/episode_{episode}/")

except KeyboardInterrupt:
    print("Simulator Interrupted")
    pass

env.close()

