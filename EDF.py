#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/16

import numpy as np


def GlobalEDF(task, processor):
    deadline = np.zeros(len(task))
    action = np.zeros(len(task))
    for i in range(len(task)):
        if task[i].isArrive:
            deadline[i] = task[i].reDeadline
        else:
            deadline[i] = np.inf
    exec_task = np.argsort(deadline)
    for i in range(processor):
        if task[exec_task[i]].isArrive:
            action[exec_task[i]] = 1
    return action

