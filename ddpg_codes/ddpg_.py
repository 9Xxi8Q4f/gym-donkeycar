import xer as buffer
import numpy as np
import keras
from keras.layers import Dense, Flatten, Concatenate, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import HeUniform
from keras.utils import plot_model
import tensorflow as tf
import json
tf.random.set_seed(1)

class Actor():

    def __init__(self, input1_shape = None, input2_shape = None,
                 fc1_dims = None, fc2_dims = None,
                 fc3_dims = None, fc4_dims = None, 
                 fc5_dims = None, n_actions = None):
        super(Actor, self).__init__()

        self.input1 = keras.layers.Input(shape=input1_shape)
        self.input2 = keras.layers.Input(shape=input2_shape)

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.init_relu = HeUniform(seed = 1)

        self.main = self.build()
        self.target = self.build()
        self.target.set_weights(self.main.get_weights())

    def build(self):

        # input1 = Flatten()(self.input1)

        # fc1 = Dense(self.fc1_dims, activation='relu', kernel_initializer= self.init_relu)(input1)
        # fc2 = Dense(self.fc1_dims, activation='relu', kernel_initializer= self.init_relu)(fc1)
        # fc2 = Dense(self.fc1_dims, activation='relu', kernel_initializer= self.init_relu)(fc2)

        fc3 = Dense(units= self.fc3_dims, activation='relu', kernel_initializer= self.init_relu)(self.input2)

        # flatten1 = Flatten()(fc2)
        # flatten2 = Flatten()(fc3)
        # conc = Concatenate()([flatten1, flatten2])

        fc4 = Dense(units= self.fc4_dims, activation='relu', kernel_initializer= self.init_relu)(fc3)
        fc5 = Dense(units= self.fc4_dims, activation='relu', kernel_initializer= self.init_relu)(fc4)
        out = Dense(units= self.n_actions, activation = "tanh")(fc5)

        model = Model(inputs = [self.input1, self.input2], outputs = out)

        return model

    def save(self, path):
        self.main.save_weights(path + '/Actor/_main/main')
        self.target.save_weights(path + '/Actor/_target/target')
    
    def load(self, path):
        self.main.load_weights(path + '/Actor/_main/main')
        self.target.load_weights(path + '/Actor/_target/target')

class Critic():

    def __init__(self, input1_shape = None, input2_shape = None,
                 fc1_dims = None, fc2_dims = None,
                 fc3_dims = None, fc4_dims = None, 
                 fc5_dims = None, n_actions = None):
        super(Critic, self).__init__()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.init_relu = HeUniform(seed = 1)
        self.input1 = keras.layers.Input(shape=input1_shape)
        self.input2 = keras.layers.Input(shape=input2_shape)
        self.input3 = keras.layers.Input(shape=(self.n_actions,))

        self.main = self.build()
        self.target = self.build()
        self.target.set_weights(self.main.get_weights())

    def build(self):

        # input1 = Flatten()(self.input1)

        # fc1 = Dense(self.fc1_dims, activation='relu', kernel_initializer= self.init_relu)(input1)
        # fc2 = Dense(self.fc1_dims, activation='relu', kernel_initializer= self.init_relu)(fc1)
        # fc2 = Dense(self.fc1_dims, activation='relu', kernel_initializer= self.init_relu)(fc2)
        
        fc3 = Dense(units= self.fc3_dims, activation='relu', kernel_initializer= self.init_relu)(self.input2)
        fc5 = Dense(units= self.fc5_dims, activation='relu', kernel_initializer= self.init_relu)(self.input3)

        # flatten1 = Flatten()(fc2)
        flatten2 = Flatten()(fc3)
        flatten3 = Flatten()(fc5)

        conc = Concatenate()([flatten2, flatten3])
        fc4 = Dense(units= self.fc4_dims, activation='relu', kernel_initializer= self.init_relu)(conc)
        out = Dense(1)(fc4)

        model = Model(inputs = [self.input1, self.input2, self.input3], outputs = out)

        return model

    def save(self, path):
        self.main.save_weights(path + '/Critic/_main/main')
        self.target.save_weights(path + '/Critic/_target/target')

    def load(self, path):
        self.main.load_weights(path + '/Critic/_main/main')
        self.target.load_weights(path + '/Critic/_target/target')

class DDPGAgent:

    def __init__(self, alpha = None, beta = None, gamma = None, obs_shape = None,
                 info_shape = None, batch_size = None, tau = None, noise = None,
                 mem_size = None, min_mem_size = None, min_action = None, optimizer = None,
                 fc1_dims = None, fc2_dims = None, max_action = None, n_actions = None,
                 fc3_dims = None, fc4_dims = None, fc5_dims = None):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        if optimizer == None:
            self.optimizer = Adam
        else: self.optimizer = optimizer
        self.batch_size = batch_size
        self.noise = noise
        self.obs_shape = obs_shape
        self.info_shape = info_shape
        self.mem_size = mem_size
        self.min_mem_size = min_mem_size
        self.max_action = max_action
        self.min_action = min_action
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.loss = 'mse'
        self.metrics = ["accuracy"]
        self.epsilon = 1.0

        self.memory = buffer.ReplayBuffer(max_size = self.mem_size, info_shape = self.info_shape,
            min_size = self.min_mem_size, n_actions= self.n_actions, input_shape = self.obs_shape)

        self.Actor = Actor(input1_shape = self.obs_shape, input2_shape = self.info_shape,
                 fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims,
                 fc3_dims = self.fc3_dims, fc4_dims = self.fc4_dims, 
                 fc5_dims = self.fc5_dims, n_actions = self.n_actions)

        self.Critic = Critic(input1_shape = self.obs_shape, input2_shape = self.info_shape,
                 fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims,
                 fc3_dims = self.fc3_dims, fc4_dims = self.fc4_dims, 
                 fc5_dims = self.fc5_dims, n_actions = self.n_actions)
        
        plot_model(self.Actor.main, to_file='model_ddpg_actor.png')
        plot_model(self.Critic.main, to_file='model_ddpg_critic.png')

        self.actor_optimizer = Adam(self.alpha)
        self.critic_optimizer = Adam(self.beta)

    def save_model(self, path):
        self.Actor.save(path)
        self.Critic.save(path)

    def load_model(self, path):
        self.Actor.load(path)
        self.Critic.load(path)

    def remember(self, state, action, reward, new_state, done, info, new_info):
        self.memory.store_transition(state, action, reward, new_state, info, new_info, done)

    def get_action(self, observation, info, evaluate = False):

        if np.random.random() < self.epsilon:
            actions = np.random.uniform(-1.0,1.0,2)
            actions = np.array([actions,actions])
        else:
            observation = tf.convert_to_tensor([observation], dtype=tf.float32)
            info = tf.convert_to_tensor([info], dtype=tf.float32)

            actions = self.Actor.main([observation, info])
            if not evaluate:
                actions += tf.random.normal(shape=[self.n_actions],
                        mean = 0.0, stddev = self.noise)
        
            actions = tf.clip_by_value(actions, self.min_action,
                            self.max_action)
        
        return actions[0]

    def learn(self):

        if (self.memory.total) < self.min_mem_size:
            return

        state, action, reward, new_state, info, new_info, done = \
                            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype = tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype = tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        infos = tf.convert_to_tensor(info, dtype=tf.float32)
        infos_ = tf.convert_to_tensor(new_info, dtype=tf.float32)
        dones = tf.convert_to_tensor(done, dtype=tf.float32)

        target_action = self.Actor.target([states_, infos_])
        target_q = self.Critic.target([states_, infos_, target_action])
        target_q = rewards + self.gamma * target_q * (1 - dones)

        #* Train the critic network
        with tf.GradientTape() as tape:
            q_values = self.Critic.main([states, infos, actions])
            critic_loss = tf.math.reduce_mean(tf.math.square(q_values - target_q))

        critic_grads = tape.gradient(critic_loss, self.Critic.main.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.Critic.main.trainable_variables))

        #* Train the actor network
        with tf.GradientTape() as tape:
            policy_action = self.Actor.main([states, infos])
            actor_loss = -tf.math.reduce_mean(self.Critic.main([states, infos, policy_action]))

        actor_grads = tape.gradient(actor_loss, self.Actor.main.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.Actor.main.trainable_variables))

        # Update the target networks
        self.update_target_networks()

        self.epsilon = self.epsilon * 0.999 if self.epsilon >0.02 else 0.02

    def update_target_networks(self):

        # Update the target actor network
        actor_weights = self.Actor.main.get_weights()
        target_actor_weights = self.Actor.target.get_weights()
        for i in range(len(actor_weights)):
            target_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * target_actor_weights[i]
        self.Actor.target.set_weights(target_actor_weights)
        
        # Update the target critic network
        critic_weights = self.Critic.main.get_weights()
        target_critic_weights = self.Critic.target.get_weights()
        for i in range(len(critic_weights)):
            target_critic_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * target_critic_weights[i]
        self.Critic.target.set_weights(target_critic_weights)

    def save_params(self, episode, episode_reward, av_reward, av_err, path):

        dictionary = {"episode" : episode, "mem_cntr" : self.memory.mem_cntr,
                "mem_cnt_" : self.memory.mem_cntr_, "total" : self.memory.total,
                "epsilon" : self.epsilon}

        parameters = "params.json"
        rew = "ep_rewards.json"
        av_rew = "av_reward.json"
        err = "av_error.json"

        with open(path + format(parameters) , 'x') as outfile:
            json.dump(dictionary, outfile)
        outfile.close()

        with open(path + format(rew), 'x') as outfile:
            json.dump(episode_reward, outfile, indent=2)
        outfile.close()

        with open(path + format(av_rew), 'x') as outfile:
            json.dump(av_reward, outfile, indent=2)
        outfile.close()

        # with open(path + format(err), 'x') as outfile:
        #     json.dump(av_err, outfile, indent=2)
        # outfile.close()

        self.memory.save_data()

    def load_params(self, path):

        parameters = "params.json"
        rew = "ep_rewards.json"
        av_rew = "av_reward.json"
        err = "av_error.json"            

        params = json.load(open(path+format(parameters)))
        self.memory.mem_cntr = params["mem_cntr"]
        self.memory.mem_cntr_ = params["mem_cnt_"]
        self.memory.total = params["total"]
        episode = params["episode"]
        self.epsilon = params["epsilon"]

        with open(path + format(rew), 'r') as outfile:
            ep_rewards = json.load(outfile)
        outfile.close()

        with open(path + format(av_rew), 'r') as outfile:
            av_reward = json.load(outfile)
        outfile.close()

        # with open(path + format(err), 'r') as outfile:
        #     av_error = json.load(outfile)
        # outfile.close()

        self.memory.load_data()

        return episode, ep_rewards, av_reward, #av_error

