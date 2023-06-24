import gym
import numpy as np
from agent import Agent
import gym_donkeycar
import time
import my_cv
 
if __name__ == '__main__':

    path_to_app = "/home/tinrafiq/Documents/DonkeySimLinux/donkey_sim.x86_64"
    port = 9091
    headless = False
    conf = { "exe_path" : path_to_app, "port" : port}

    env = gym.make("donkey-generated-roads-v0", conf = conf)
    time.sleep(5)
    print("Environment is setting up.")

    observation, reward, done, info = env.reset()
    observation = my_cv.process_image(observation)

    agent = Agent(input_dims=observation.shape,
                  n_actions=2)
    
    n_games = 250
    # figure_file = 'plots/pendulum.png'
    best_score = -1500.0
    score_history = []
    load_chekpoint = False
    
    if load_chekpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation,action,reward,observation_,done)
            n_steps += 1
        agent.learn()
        agent.load_model()
        evaluate = True
    else: evaluate = False

    for i in range(n_games):
        observation, reward, done, info = env.reset()

        done = False
        score = 0
        step = 0
        observation = my_cv.process_image(observation)

        while not done:

            action = agent.choose_action(observation, evaluate)
            action = np.array(action,dtype=np.float32)
            action[1] = (action[1] + 1.5) / 20.0 

            observation_, reward, done, info_ = env.step(action)
            observation_ = my_cv.process_image(observation_)

            score +=reward
            agent.remember(observation,action,reward,observation_,done)
            if not load_chekpoint:
                agent.learn()
            
            observation = observation_
            step +=1

        # for i in range(step):
        #     agent.learn()
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_chekpoint:
                agent.save_model()

        print('episode ', i, 'score %.1f' % score,
               'avg score %.1f' % avg_score)
        
    # if not load_chekpoint:
    #     x = [i+1 for i in range(n_games)]
    #     plot_learning_curve(x, score_history, figure_file)

