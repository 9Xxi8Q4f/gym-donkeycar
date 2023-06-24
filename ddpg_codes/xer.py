import random
import numpy as np

class ReplayBuffer(object):
    """
    * init the values
    * for DQN actions are discrete
    """
    def __init__(self, max_size, min_size, input_shape, n_actions, info_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.mem_cntr_ = 0
        self.total = 0
        self.min_size = min_size
        
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.info = np.zeros((self.mem_size, *info_shape))
        self.new_info = np.zeros((self.mem_size, *info_shape))

    def store_transition(self, state, action, reward, state_, info, new_info, done):

        if self.mem_size > self.mem_cntr:
            index = self.mem_cntr % self.mem_size
            self.mem_cntr += 1

        elif self.mem_size == self.mem_cntr:
            index = self.mem_cntr_ % self.mem_size
            self.mem_cntr_ += 1
            if self.mem_cntr_ == self.mem_cntr:
                self.mem_cntr_ = 0

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.info[index] = info
        self.new_info[index] = new_info

        self.total +=1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        infos = self.info[batch]
        new_infos = self.new_info[batch]

        return states, actions, rewards, states_, infos, new_infos, dones

    def save_data(self):
        np.save("ddpg/data/states.npy", self.state_memory[:self.mem_cntr])
        np.save("ddpg/data/states_.npy", self.new_state_memory[:self.mem_cntr])
        np.save("ddpg/data/actions.npy", self.action_memory[:self.mem_cntr])
        np.save("ddpg/data/rewards.npy", self.reward_memory[:self.mem_cntr])
        np.save("ddpg/data/terminal.npy", self.terminal_memory[:self.mem_cntr])
        np.save("ddpg/data/info.npy", self.info[:self.mem_cntr])
        np.save("ddpg/data/info_.npy", self.new_info[:self.mem_cntr])

    def load_data(self):
        self.state_memory[:self.mem_cntr] = np.load("ddpg/data/states.npy")
        self.new_state_memory[:self.mem_cntr] = np.load("ddpg/data/states_.npy")
        self.action_memory[:self.mem_cntr] = np.load("ddpg/data/actions.npy")
        self.reward_memory[:self.mem_cntr] = np.load("ddpg/data/rewards.npy")
        self.terminal_memory[:self.mem_cntr] = np.load("ddpg/data/terminal.npy")
        self.info[:self.mem_cntr] = np.load("ddpg/data/info.npy")
        self.new_info[:self.mem_cntr] = np.load("ddpg/data/info_.npy")


