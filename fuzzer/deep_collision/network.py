import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from loguru import logger
from omegaconf import DictConfig

HyperParameter = dict(BATCH_SIZE=32, GAMMA=0.9, EPS_START=1, EPS_END=0.1, EPS_DECAY=6000, TARGET_UPDATE=100,
                      lr=1e-2, INITIAL_MEMORY=2000, MEMORY_SIZE=2000)

N_STATES = 12 # ego_x, ego_y, ego_z, rain, fog, wetness, timeofday, signal, rx (rotation), ry, rz, speed
N_ACTIONS = 5 # Left_Lane, Right_Lane, Current_Lane # now only support in our sim platform
ENV_A_SHAPE = 0

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 200)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(200, 200)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(200, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        # x = self.fc1(x)
        # print('x1', x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # print('x', x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):

    cfg: DictConfig

    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((HyperParameter['MEMORY_SIZE'], N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=HyperParameter['lr'])
        self.loss_func = nn.MSELoss()
        self.steps_done = 0

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        eps_threshold = HyperParameter['EPS_END'] + (
                HyperParameter['EPS_START'] - HyperParameter['EPS_END']) * math.exp(
            -1. * self.steps_done / HyperParameter['EPS_DECAY'])
        self.steps_done += 1
        if np.random.uniform() < eps_threshold:  # greedy
            actions_value = self.eval_net.forward(x)
            # print('action value: ', actions_value, actions_value.shape)
            action = torch.max(actions_value, 1)[1].data.numpy()
            # print(actions_value.data)
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
            # print(action)
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % HyperParameter['MEMORY_SIZE']
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % HyperParameter['TARGET_UPDATE'] == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(HyperParameter['MEMORY_SIZE'], HyperParameter['BATCH_SIZE'])
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + HyperParameter['GAMMA'] * q_next.max(1)[0].view(HyperParameter['BATCH_SIZE'], 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logger.debug('optimize DRL')