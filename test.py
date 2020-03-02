#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/9
from ddpg import *
from task import *
import copy
t =[]
e = 0
d = 0
for i in range(10000):
    t = Task()
    e += t.execute_time
    d += t.deadline
print(e/1000, d/1000)
task1 = Task()
task2 = Task()
task3 = Task()
# task3.arrive()
task = [task1, task2, task3]
t22 = [task[1], task[2]]


