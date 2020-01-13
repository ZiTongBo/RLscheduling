#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/10

GPU = False

device_idx = 0
state_dim = 8
action_dim = 1

MAX_PERIOD = 30
MIN_PERIOD = 20
MAX_EXECUTE_TIME = 8
MIN_EXECUTE_TIME = 3
MAX_DEADLINE = 20
MIN_DEADLINE = 10

hidden_dim = 512
explore_steps = 20  # for random exploration
batch_size = 64
replay_buffer_size = 1e6
model_path = './model/'
max_episodes = 1000
max_steps = 1000