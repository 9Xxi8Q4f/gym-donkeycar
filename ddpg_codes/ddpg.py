import xer as buffer
import numpy as np
import keras
from keras.layers import Dense, Flatten, Concatenate #, Conv2D
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
                 fc5_dims = None, n_actions = None, learning_rate = None,
                 optimizer = None, loss = None, metrics = None):
        super(Actor, self).__init__()

        self.input1 = keras.layers.Input(shape=input1_shape)
        self.input2 = keras.layers.Input(shape=input2_shape)

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.init_relu = HeUniform(seed = 1)

        self.main = self.build()
        self.target = self.build()

        self.main.compile(optimizer = optimizer(learning_rate), loss = self.loss, metrics = self.metrics)
        self.target.compile(optimizer = optimizer(learning_rate), loss = self.loss, metrics = self.metrics)

    def build(self):

        fc1 = Dense(units= self.fc1_dims, activation='relu', kernel_initializer= self.init_relu)(self.input1)
        fc2 = Dense(units= self.fc2_dims, activation='relu', kernel_initializer= self.init_relu)(fc1)
        fc3 = Dense(units= self.fc3_dims, activation='relu', kernel_initializer= self.init_relu)(self.input2)
        flatten1 = Flatten()(fc2)
        flatten2 = Flatten()(fc3)
        conc = Concatenate()([flatten1, flatten2])
        fc4 = Dense(units= self.fc4_dims, activation='relu', kernel_initializer= self.init_relu)(conc)
        out = Dense(units= self.n_actions, activation = "tanh")(fc4)

        model = Model(inputs = [self.input1, self.input2], outputs = out)

        return model

    def save(self, path):
        self.main.save_weights(path + '/Actor/_main')
        self.target.save_weights(path + '/Actor/_target')

class Critic():

    def __init__(self, input1_shape = None, input2_shape = None,
                 fc1_dims = None, fc2_dims = None,
                 fc3_dims = None, fc4_dims = None, 
                 fc5_dims = None, n_actions = None, learning_rate = None,
                 optimizer = None, loss = None, metrics = None):
        super(Critic, self).__init__()

        self.learning_rate = learning_rate
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.init_relu = HeUniform(seed = 1)
        self.input1 = keras.layers.Input(shape=input1_shape)
        self.input2 = keras.layers.Input(shape=input2_shape)
        self.input3 = keras.layers.Input(shape=(self.n_actions,))

        self.main = self.build()
        self.target = self.build()

        self.main.compile(optimizer = optimizer(learning_rate), loss = loss, metrics = metrics)
        self.target.compile(optimizer = optimizer(learning_rate), loss = loss, metrics = metrics)

    def build(self):

        fc1 = Dense(units= self.fc1_dims, activation='relu', kernel_initializer= self.init_relu)(self.input1)
        fc2 = Dense(units= self.fc2_dims, activation='relu', kernel_initializer= self.init_relu)(fc1)
        fc3 = Dense(units= self.fc3_dims, activation='relu', kernel_initializer= self.init_relu)(self.input2)
        fc5 = Dense(units= self.fc5_dims, activation='relu', kernel_initializer= self.init_relu)(self.input3)
        flatten1 = Flatten()(fc2)
        flatten2 = Flatten()(fc3)
        flatten3 = Flatten()(fc5)
        conc = Concatenate()([flatten1, flatten2, flatten3])
        fc4 = Dense(units= self.fc4_dims, activation='relu', kernel_initializer= self.init_relu)(conc)
        out = Dense(1)(fc4)

        model = Model(inputs = [self.input1, self.input2, self.input3], outputs = out)

        return model

    def save(self, path):
        self.main.save_weights(path + '/Critic/_main')
        self.target.save_weights(path + '/Critic/_target')

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

        self.memory = buffer.ReplayBuffer(max_size = self.mem_size, info_shape = self.info_shape,
            min_size = self.min_mem_size, n_actions= self.n_actions, input_shape = self.obs_shape)

        self.actor = Actor(input1_shape = self.obs_shape, input2_shape = self.info_shape,
                 fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims,
                 fc3_dims = self.fc3_dims, fc4_dims = self.fc4_dims, 
                 fc5_dims = self.fc5_dims, n_actions = self.n_actions,
                 learning_rate= self.alpha, optimizer = self.optimizer,
                 loss = self.loss, metrics = self.metrics)

        self.critic = Critic(input1_shape = self.obs_shape, input2_shape = self.info_shape,
                 fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims,
                 fc3_dims = self.fc3_dims, fc4_dims = self.fc4_dims, 
                 fc5_dims = self.fc5_dims, n_actions = self.n_actions,
                 learning_rate= self.beta, optimizer = self.optimizer,
                 loss = self.loss, metrics = self.metrics)

        plot_model(self.actor.main, to_file='model_ddpg_actor.png')
        plot_model(self.critic.main, to_file='model_ddpg_critic.png')

    def save_model(self, path):
        self.actor.save(path)
        self.critic.save(path)

    def remember(self, state, action, reward, new_state, done, info, new_info):
        self.memory.store_transition(state, action, reward, new_state, info, new_info, done)

    def get_action(self, observation, info, evaluate = False):
        observation = tf.convert_to_tensor([observation], dtype=tf.float32)
        info = tf.convert_to_tensor([info], dtype=tf.float32)

        actions = self.actor.main([observation, info])
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions],
                    mean = 0.0, stddev = self.noise)
        
        actions = tf.clip_by_value(actions, self.min_action,
                        self.max_action)
        
        return actions[0]

    def update_network_parameters(self, tau = None):
            """
            * soft update: try hard update?
            """
            
            if tau is None:
                tau = self.tau

            weights = []
            targets = self.actor.target.weights
            for i, weight in enumerate(self.actor.main.weights):
                weights.append(weight * tau + targets[i]*(1-tau))
            self.actor.target.set_weights(weights)

            weights = []
            targets = self.critic.target.weights
            for i, weight in enumerate(self.critic.main.weights):
                weights.append(weight * tau + targets[i]*(1-tau))
            self.critic.target.set_weights(weights)

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

        with tf.GradientTape() as tape:
            target_actions = self.actor.target([states_, infos_])
            critic_value_ = tf.squeeze(self.critic.target([states_,infos_, 
                                                        target_actions]),1)

            critic_value = tf.squeeze(self.critic.main([states, infos, actions]),1)

            target = rewards + self.gamma * critic_value_ * (1-done)

            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                        self.critic.main.trainable_variables)

        self.critic.main.optimizer.apply_gradients(zip(critic_network_gradient,
                        self.critic.main.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor.main([states,infos])
            actor_loss = -self.critic.main([states, infos, new_policy_actions])
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                        self.actor.main.trainable_variables)
        self.actor.main.optimizer.apply_gradients(zip(actor_network_gradient, 
                            self.actor.main.trainable_variables))

        # self.update_network_parameters()     

        if self.memory.total % 1000 == 0:
            self.actor.target.set_weights(self.actor.main.get_weights())
            self.critic.target.set_weights(self.critic.main.get_weights())

    def save_params(self, episode, episode_reward, av_reward, av_err, path):
        dictionary = {"episode" : episode, "mem_cntr" : self.memory.mem_cntr,
                "mem_cnt_" : self.memory.mem_cntr_, "total" : self.memory.total }

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
