#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/9

from ddpg import *




# 在 sort 函数中排序字段

dt = np.dtype([('no',  'S10'), ('age',  float)])
actions = np.array([], dtype=dt)

actions = np.append(actions,np.array([(str(0), 22)], dtype = dt))
actions = np.append(actions,np.array([(str(1), 12)], dtype = dt))
ac = np.array([2,1,0])
print(ac+1)
print(np.array(ac).reshape(-1,1))
print(np.argsort(ac)[::-1])
print ('我们的数组是：')
print (actions)
print ('\n')
print ('按 name 排序：')
print (float(np.sort(actions, order='age')[0][1]))