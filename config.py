#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/10

GPU = True

DEVICE_INDEX = 0
STATE_DIM = 9
ACTION_DIM = 1

MAX_PERIOD = 30
MIN_PERIOD = 20

MAX_DEADLINE = 40
MIN_DEADLINE = 10
GRANULARITY = 1 / 2
MAX_EXECUTE_TIME = MAX_DEADLINE
MIN_EXECUTE_TIME = 5
LAMBDA = 100
FREQUENCY = 20
NO_TASK = 50
NO_PROCESSOR = 8
TASK_PER_PROCESSOR = 15

HIDDEN = [32, 16]
EXPLORATION_EPISODES = 0  # for random exploration
BATCH_SIZE = 16
BUFFER_SIZE = 1e3
MODEL_PATH = './model/'
MAX_EPISODES = 100000
MAX_STEPS = 10000

DEBUG = False
EDF = True
