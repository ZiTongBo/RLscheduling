#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/9

import numpy as np

np.random.seed(199686)


# TODO: 生成任务集
class Task:
    def __init__(self):
        self.executeTime = 2
        self.period = 4
        self.deadline = 4
        self.count = 5
        self.isArrive = False

    def arrive(self):
        self.isArrive = True
        self.executeTime = 1
        self.deadline = 5

    def execute(self):
        self.executeTime -= 1
        if self.executeTime == 0:
            self.count -= 1
            self.isArrive = False

        return not self.isArrive

    def miss(self):
        self.count -= 1
        self.isArrive = False


