#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/9

import numpy as np
from config import *

np.random.seed(6786790)


class Task:
    def __init__(self):
        interval = np.random.poisson(LAMBDA, FREQUENCY)
        self.arrive_time = []
        for i in range(FREQUENCY):
            interval_from_beginning = 0
            for j in range(i + 1):
                interval_from_beginning += interval[j]
            self.arrive_time.append(interval_from_beginning)
        self.deadline = np.random.randint(500, 1500, 1)[0]
        self.execute_time = round(np.random.exponential(self.deadline/6))
        while self.execute_time > self.deadline:
            self.execute_time = round(np.random.exponential(self.deadline/6))
        self.execute_time = np.clip(self.execute_time, MIN_EXECUTE_TIME, MAX_EXECUTE_TIME)
        self.count = 0

    def create_instance(self):
        i = Instance(self)
        self.count += 1
        return i


class Instance:
    def __init__(self, task):
        self.execute_time = task.execute_time
        self.deadline = task.deadline
        self.laxity_time = self.deadline - self.execute_time
        self.over = False

    def step(self, execute):
        if execute:
            self.execute_time -= 1
        self.deadline -= 1
        self.laxity_time = self.deadline - self.execute_time
        if self.execute_time == 0:
            return "finish"
        if self.deadline == 0:
            return "miss"
        return 0

