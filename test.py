#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/9

from ddpg import *
from task import *
import copy

def GlobalEDF(task, processor):
    deadline = np.zeros(len(task))
    action = np.zeros(len(task))
    for i in range(len(task)):
        if task[i].isArrive:
            deadline[i] = task[i].reDeadline
        else:
            deadline[i] = np.inf
    exec_task = np.argsort(deadline)
    print(deadline)
    print(exec_task)
    for i in range(processor):
        if task[exec_task[i]].isArrive:
            action[exec_task[i]] = 1
    return action
# 在 sort 函数中排序字段

dt = np.dtype([('no',  'S10'), ('age',  float)])
actions = np.array([], dtype=dt)

actions = np.append(actions,np.array([(str(0), 22)], dtype = dt))
actions = np.append(actions,np.array([(str(1), 12)], dtype = dt))
ac = np.array([[0,0,0],[3,2,1]])
print(ac[:, 0])
task1 = Task()
task2 = Task()
task3 = Task()
task1.arrive()
task2.arrive()
# task3.arrive()
task = [task1, task2, task3]
t2=copy.deepcopy(task)
# for i in range(3):
#     t2[i] = copy.deepcopy(task[i])
print(task[0].executeTime)
task[0].executeTime = 111
print(t2[0].executeTime)
print(GlobalEDF(task, 2))
print(ac+1)
print(np.array(ac).reshape(-1,1))
print(np.argsort(ac)[::-1])

