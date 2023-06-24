import os 
import tensorflow as tf
import keras
from keras.layers import Dense, Convolution2D, Flatten
import numpy as np
import tensorflow as tf
import keras
from keras.optimizers import Adam
import my_cv
import time
import gym_donkeycar
from collections import deque
import skimage as skimage
import random
import numpy as np
import cv2
import gym
from buffer import ReplayBuffer
 
EPISODES = 10000
img_rows , img_cols = 80, 80
# Convert image into Black and white
img_channels = 4 # We stack 4 frames

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims = 512, fc2_dims = 512,
                 name = 'critic', chkpt_dir = 'tmp/ddpg'):
        super(CriticNetwork, self).__init__()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name + '_ddpg.h5')
        
        self.fc1 = Convolution2D(32, (8, 8), strides=(4,4), activation='relu', padding="same",input_shape=(img_rows,img_cols,img_channels))
        self.fc2 = Convolution2D(64, (4, 4), strides=(2,2), activation='relu', padding="same")
        self.fc3 = Convolution2D(64, (3, 3), strides=(1,1), activation='relu', padding="same")
        self.fc4 = Dense(512, activation='relu')
        self.q = Dense(1,activation=None)
        self.flatten = Flatten()

    def call(self, state, action):

        observation_value = self.fc1(state)
        observation_value = self.fc2(observation_value)
        observation_value = self.fc3(observation_value)
        observation_value = self.flatten(observation_value)

        action_value = self.fc4(tf.concat([observation_value, action], axis = 1))

        q = self.q(action_value)

        return q
    
class ActorNetwork(keras.Model):
    def __init__(self, n_actions=2, fc1_dims = 512, fc2_dims = 512,
                 name = 'actor', chkpt_dir = 'tmp/ddpg'):
        super(ActorNetwork, self).__init__()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name + '_ddpg.h5')
        
        self.fc1 = Convolution2D(32, (8, 8), strides=(4,4), activation='relu', padding="same",input_shape=(img_rows,img_cols,img_channels))
        self.fc2 = Convolution2D(64, (4, 4), strides=(2,2), activation='relu', padding="same")
        self.fc3 = Convolution2D(64, (3, 3), strides=(1,1), activation='relu', padding="same")
        self.fc4 = Dense(512, activation='relu')
        self.flatten = Flatten()
        self.mu = Dense(self.n_actions ,activation='tanh')

    def call(self, state):

        prob = self.fc1(state)
        prob = self.fc2(prob)
        prob = self.fc3(prob)
        prob = self.flatten(prob)
        prob = self.fc4(prob)

        mu = self.mu(prob)

        return mu
        
class Agent:
    def __init__(self, alpha = 0.001, input_dims = None,
                 beta = 0.002, gamma = .99,
                 n_actions = 2,
                 tau = 0.005, max_size = 10000,
                 batch_size = 64, noise =0.1, train_start = 64):

        self.lane_detection = True # Set to True to train on images with segmented lane lines

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = 1.0
        self.min_action = -1.0
        self.t = 0
        self.train_start = train_start

        self.actor = ActorNetwork(n_actions = n_actions, name = 'actor')
        self.critic = CriticNetwork(name = 'critic')
        self.target_actor = ActorNetwork(n_actions=n_actions,
                                         name = 'target_actor')
        self.target_critic = CriticNetwork(name = 'target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

        # Create replay memory using deque
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

    def process_image(self, obs):

        if not self.lane_detection:
            obs = skimage.color.rgb2gray(obs)
            obs = skimage.transform.resize(obs, (img_rows, img_cols))
            return obs
        else:
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = cv2.resize(obs, (img_rows, img_cols))
            edges = my_cv.detect_edges(obs, low_threshold=50, high_threshold=150)


            rho = 0.8
            theta = np.pi/180
            threshold = 25
            min_line_len = 5
            max_line_gap = 10

            hough_lines = my_cv.hough_lines(edges, rho, theta, threshold, min_line_len, max_line_gap)

            left_lines, right_lines = my_cv.separate_lines(hough_lines)

            filtered_right, filtered_left = [],[]
            if len(left_lines):
                filtered_left = my_cv.reject_outliers(left_lines, cutoff=(-30.0, -0.1), lane='left')
            if len(right_lines):
                filtered_right = my_cv.reject_outliers(right_lines,  cutoff=(0.1, 30.0), lane='right')

            lines = []
            if len(filtered_left) and len(filtered_right):
                lines = np.expand_dims(np.vstack((np.array(filtered_left),np.array(filtered_right))),axis=0).tolist()
            elif len(filtered_left):
                lines = np.expand_dims(np.expand_dims(np.array(filtered_left),axis=0),axis=0).tolist()
            elif len(filtered_right):
                lines = np.expand_dims(np.expand_dims(np.array(filtered_right),axis=0),axis=0).tolist()

            ret_img = np.zeros((80,80))

            if len(lines):
                try:
                    my_cv.draw_lines(ret_img, lines, thickness=1)
                except:
                    pass

            return ret_img

    def update_network_parameters(self, tau=None):

        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i,weight in enumerate(self.critic.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state,action,reward,new_state,done)

    def choose_action(self,observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape =[self.n_actions],
                                         mean= 0.0, stddev = self.noise)

        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]

    def learn(self):
        if len(self.memory) < self.train_start:
            return

        state, action, reward, new_state, done = \
        self.memory.sample_bufer(self.batch_size)

        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)
        states = tf.convert_to_tensor(state_t, dtype=np.float32)
        states_ = tf.convert_to_tensor(state_t1, dtype = np.float32)
        actions = tf.convert_to_tensor(action, dtype=np.float32)
        rewards = reward

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(
                states_, target_actions),1)
            
            critic_value = tf.squeeze(self.critic(states,actions),1)
            target = rewards + self.gamma*critic_value_*(1-done)

            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))
        
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))
        
        self.update_network_parameters()

def run_ddpg():

    path_to_app = "/home/tinrafiq/Documents/DonkeySimLinux/donkey_sim.x86_64"
    port = 9091
    conf = { "exe_path" : path_to_app, "port" : port }


    env = gym.make("donkey-generated-roads-v0", conf = conf)
    time.sleep(5)
    print("Environment is setting up.")

    agent = Agent(n_actions=1)
    throttle = 0.1 # Set throttle as constant value

    n_games = 250
    # figure_file = 'plots/pendulum.png'
    best_score = -1000.0
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
            score = 0
            step = 0

            x_t = agent.process_image(observation)

            s_t = np.stack((x_t,x_t,x_t,x_t),axis=2)
            # In Keras, need to reshape
            s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2]) #1*80*80*4       
            ep_reward = 0

            while not done:
                steering = agent.choose_action(s_t, evaluate)
                action = np.array([steering[0], throttle])

                next_obs, reward, done, info = env.step(action)

                x_t1 = agent.process_image(next_obs)

                x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
                s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3) #1x80x80x4

                agent.replay_memory(s_t, steering, reward, s_t1, done)

                agent.learn()
                s_t = s_t1
                agent.t = agent.t + 1
                step +=1

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                # if not load_chekpoint:
                #     agent.save_model()
            if i%20 == 0:
                print('episode ', i, ' episode len ', step,
                      'score %.1f' % score,
                      'avg score %.1f' % avg_score)


if __name__ == "__main__":

    run_ddpg()