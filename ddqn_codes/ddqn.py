import per as buffer
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

class qNetwork(keras.Model):
 
    def __init__(self, input1_shape = None, input2_shape = None,
                 fc1_dims = None, fc2_dims = None,
                 fc3_dims = None, fc4_dims = None, 
                 fc5_dims = None, n_actions = None, learning_rate= None):
        super(qNetwork, self).__init__()

        self.input1 = keras.layers.Input(shape=input1_shape)
        self.input2 = keras.layers.Input(shape=input2_shape)

        self.learning_rate = learning_rate
        # self.fc1_dims = fc1_dims
        # self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        # self.fc3_dims = fc3_dims
        # self.fc4_dims = fc4_dims
        # self.fc5_dims = fc5_dims
        self.init_relu = HeUniform(seed = 1)
 
    def build(self):
        fc1 = Conv2D(24, (5, 5), strides=(2, 2), padding="same", activation='relu')(self.input1)
        fc2 = Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation='relu')(fc1)
        fc3 = Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation='relu')(fc2)
        fc4 = Flatten()(fc3)
        fc5 = Dense(units=512, activation='relu')(fc4)
        out = Dense(units=self.n_actions)((fc5))
        
        model = Model(inputs = [self.input1, self.input2], outputs = out)

        return model

class DDQNAgent:

    def __init__(self, alpha = None, gamma = None, epsilon = None, obs_shape = None,
                 info_shape = None, batch_size = None, epsilon_dec = None, 
                 epsilon_end = None, mem_size = None, min_mem_size = None, 
                 replace_target = None, fc1_dims = None, fc2_dims = None,
                 fc3_dims = None, fc4_dims = None, fc5_dims = None):

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.min_mem_size = min_mem_size
        self.replace_target = replace_target
        self.obs_shape = obs_shape
        self.info_shape = info_shape
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims

        #* Action Space Setting
        steering_actions = [-1.0, -0.5, 0.0, 0.5, 1.0]
        throttle_actions = [0.0, 0.1, 0.2]
        self.discrete_action_space = np.array(np.meshgrid(steering_actions, 
                            throttle_actions)).T.reshape(-1,2)

        self.n_actions = len(self.discrete_action_space)
        self.action_space = [i for i in range(self.n_actions)]

        self.memory = buffer.ReplayBuffer(max_size = self.mem_size, info_shape = self.info_shape,
                            min_size = self.min_mem_size, discrete= True, 
                            n_actions= self.n_actions, input_shape = self.obs_shape)
        
        self.network = qNetwork(input1_shape = self.obs_shape, input2_shape = self.info_shape,
            fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims,
            fc3_dims = self.fc3_dims, fc4_dims = self.fc4_dims, fc5_dims = self.fc5_dims,
                               n_actions = self.n_actions, learning_rate = self.alpha)

        self.q_eval = self.network.build()
        self.q_target = self.network.build()
        self.q_eval.summary()

        self.q_eval.compile(optimizer = Adam(learning_rate = self.alpha),
                            loss = 'mse', metrics = ['accuracy'])

        self.q_target.compile(optimizer = Adam(learning_rate = self.alpha),
                            loss = 'mse', metrics = ['accuracy'])
        
        plot_model(self.q_eval, to_file='model_ddqn.png')

    def save_weights(self, path):
        self.q_eval.save_weights(path + '_eval')
        self.q_target.save_weights(path + '_target')

    def load_weights(self, path):
        self.q_eval.load_weights(path + '_eval')
        self.q_target.load_weights(path + '_target')

    def epsilon_decay(self):
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_end \
        else self.epsilon_end

    def remember(self, state, action, reward, new_state, done, info, new_info):
        self.memory.store_transition(state, action, reward, new_state, info, new_info, done)

    def get_action(self, observation, info):

        if np.random.random() > self.epsilon:
            observation = tf.convert_to_tensor([observation], dtype=tf.float32)
            info = tf.convert_to_tensor([info], dtype=tf.float32)

            qs_= self.q_eval.predict([observation, info])
            action_index = np.argmax(qs_)
            action = self.discrete_action_space[action_index]
        else:
            action_index = np.random.randint(0, self.n_actions)
            action = self.discrete_action_space[action_index]
        
        return action, action_index

    def train(self):

        if (self.memory.total) < self.min_mem_size:
            return

        #* and ELSE:
        #* sample minibatch and get states vs..
        state, action, reward, new_state, info, new_info, done, sample_indices = \
                            self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        #* get the q values of current states by main network
        q_pred = self.q_eval.predict([state, info])

        #! for abs error
        target_old = np.array(q_pred)

        #* get the q values of next states by target network
        q_next = self.q_target.predict([new_state, new_info]) #! target_val

        #* get the q values of next states by main network
        q_eval = self.q_eval.predict([new_state, new_info]) #! target_next

        #* get the actions with highest q values
        max_actions = np.argmax(q_eval, axis=1)

        #* we will update this dont worry
        q_target = q_pred

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        #* new_q = reward + DISCOUNT * max_future_q
        q_target[batch_index, action_indices] = reward + \
                    self.gamma*q_next[batch_index, max_actions.astype(int)]*done

        #* error
        error = target_old[batch_index, action_indices]-q_target[batch_index, action_indices]
        self.memory.set_priorities(sample_indices, error)

        #* now we fit the main model (q_eval)
        _ = self.q_eval.fit([state, info], q_target, verbose=0)

        #* If counter reaches set value, update target network with weights of main network
        #* it will update it at the very beginning also
        if self.memory.total & self.replace_target == 0:
            self.update_network_parameters()
            print("Target Updated")

        self.epsilon_decay()

    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())

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

