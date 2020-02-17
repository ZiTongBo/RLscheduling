#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/9

from ddpg import *
from task import *
import copy
from EDF import *

ac = np.array([[0,0,0],[3,2,1]])
print(round(np.random.exponential(6)))
print(ac[:, 0])
for i in range(40):
    print(i)
task1 = Task()
task2 = Task()
task3 = Task()
# task3.arrive()
task = [task1, task2, task3]
t22 = [task[1], task[2]]


