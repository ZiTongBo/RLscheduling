#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/10

GPU = True

device_idx = 0
state_dim = 6
action_dim = 1

MAX_PERIOD = 30
MIN_PERIOD = 20

MAX_DEADLINE = 180
MIN_DEADLINE = 100
GRANULARITY = 1 / 2
MAX_EXECUTE_TIME = MAX_DEADLINE
MIN_EXECUTE_TIME = 5
LAMBDA = 100
FREQUENCY = 20
NO_TASK = 50
NO_PROCESSOR = 20

HIDDEN = [32, 16]
explore_episodes = 0  # for random exploration
BATCH_SIZE = 16
replay_buffer_size = 1e4
model_path = './model/'
max_episodes = 100000
max_steps = 10000

DEBUG = False
EDF = True
