#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/9

import numpy as np

np.random.seed(199686)


# TODO: 生成任务集
class Task:
    def __init__(self):
        self.time = np.random.randint(0, 3)
        self.executeTime = np.random.randint(3, 5)
        self.deadline = self.executeTime + np.random.randint(1, 3)
        self.period = self.deadline + np.random.randint(1, 3)
        self.count = 10
        self.isArrive = False

    def arrive(self):
        self.isArrive = True
        self.reExecuteTime = self.executeTime
        self.reDeadline = self.deadline

    def execute(self):
        self.reExecuteTime -= 1
        if self.reExecuteTime == 0:
            self.count -= 1
            self.isArrive = False

    def miss(self):
        self.count -= 1
        self.isArrive = False


