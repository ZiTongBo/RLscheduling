#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/9


import math
import random
from config import *
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display

import argparse

if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
torch.autograd.set_detect_anomaly(True)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, init_w=3e-3):
        super(ActorNetwork, self).__init__()
        self.action_dim = output_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)  # output dim = dim of action

        # weights initialization
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        activation = torch.relu
        x = activation(self.linear1(state))
        x = activation(self.linear2(x))
        x = torch.sigmoid(self.linear3(x)).clone()  # for simplicity, no restriction on action rang
        return x

    def select_action(self, state, noise, noise_scale=0.5):
        '''
        select action for sampling, no gradients flow, noisy action, return .cpu
        '''
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # state dim: (N, dim of state)
        normal = Normal(0, noise)
        # print(state)
        action = self.forward(state)
        # print(action)
        noise = noise_scale * normal.sample(action.shape).to(device)
        action += noise
        action = torch.from_numpy(np.clip(action.detach().numpy(), 0, 1)[0])
        return action.detach().cpu().numpy()

    def sample_action(self, action_range=1.):
        normal = Normal(0.5, 0.4)
        random_action = torch.from_numpy(np.clip(normal.sample((1,)).numpy(), 0.001, 1))
        return random_action.cpu().numpy()

    def evaluate_action(self, state, noise_scale=0.0):
        '''
        evaluate action within GPU graph, for gradients flowing through it, noise_scale controllable
        '''
        normal = Normal(0, 1)
        action = self.forward(state)
        # action = torch.tanh(action)
        noise = noise_scale * normal.sample(action.shape).to(device)
        action += noise
        return action


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DDPG():
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_dim):
        self.replay_buffer = replay_buffer
        self.q_net = QNetwork(state_dim + action_dim, hidden_dim).to(device)
        self.target_qnet = QNetwork(state_dim + action_dim, hidden_dim).to(device)
        self.policy_net = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)

        print('Q network: ', self.q_net)
        print('Policy network: ', self.policy_net)

        for target_param, param in zip(self.target_qnet.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data)
        self.q_criterion = nn.MSELoss()
        q_lr = 1e-4
        policy_lr = 1e-5
        self.update_cnt = 0

        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

    def target_soft_update(self, net, target_net, soft_tau):
        # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net

    def update(self, batch_size, reward_scale=10.0, gamma=0.99, soft_tau=1e-2, policy_up_itr=10, target_update_delay=3,
               warmup=True):
        self.update_cnt += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predict_q = self.q_net(state, action)  # for q
        new_next_action = self.target_policy_net.evaluate_action(next_state)  # for q
        new_action = self.policy_net.evaluate_action(state)  # for policy
        predict_new_q = self.q_net(state, new_action)  # for policy
        target_q = reward + (1 - done) * gamma * self.target_qnet(next_state, new_next_action)  # for q
        # reward = reward_scale * (reward - reward.mean(dim=0)) /reward.std(dim=0) # normalize with batch mean and std

        # train qnet
        q_loss = self.q_criterion(predict_q, target_q.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # train policy_net
        policy_loss = -torch.mean(predict_new_q)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # update the target_qnet
        if self.update_cnt % target_update_delay == 0:
            self.target_qnet = self.target_soft_update(self.q_net, self.target_qnet, soft_tau)
            self.target_policy_net = self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        return q_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()

    def save_model(self, path):
        torch.save(self.q_net.state_dict(), path + '_q')
        torch.save(self.target_qnet.state_dict(), path + '_target_q')
        torch.save(self.policy_net.state_dict(), path + '_policy')

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path + '_q'))
        self.target_qnet.load_state_dict(torch.load(path + '_target_q'))
        self.policy_net.load_state_dict(torch.load(path + '_policy'))
        self.q_net.eval()
        self.target_qnet.eval()
        self.policy_net.eval()


def plot(rewards):
    plt.figure(figsize=(20, 5))
    plt.plot(rewards)
    plt.savefig('plot/ddpg.png')
    # plt.show()
    plt.clf()


def _reverse_action(action):
    low = 0
    high = 1

    action = 2 * (action - low) / (high - low) - 1
    action = np.clip(action, low, high)

    return action


class NormalizedActions():  # gym env wrapper
    def _action(self, action):
        low = 0
        high = 1

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

