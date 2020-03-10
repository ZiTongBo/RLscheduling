#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/9
from ddpg_torch import *
from task import *
import copy
import ddpg_tf
import tensorflow as tf
import ddpg_torch
import math
import torch
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
t =[]
e = 0
with tf.variable_scope('foo'):
    v = tf.get_variable('v', [1], initializer=tf.constant_initializer(1.0))


def partition(arr, low, high):
    i = (low - 1)  # 最小元素索引
    pivot = arr[high]

    for j in range(low, high):

        # 当前元素小于或等于 pivot
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return (i + 1)


# arr[] --> 排序数组
# low  --> 起始索引
# high  --> 结束索引

# 快速排序函数
def quickSort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)

        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)


d = 8
print(ddpg_tf.lcm([2,3,4]))
print(np.hstack(([2],[3])))
c = [[[2, 0, 0, 0, 0], [0, 0, 0, 0, 0],[2,3,4,3,2]], [[0], [0],[1]], 0, [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],[3,3,2,1,1]]]
cc= [[2],[3],[4]]
print(np.mean(cc[0:2],axis=0))
print(np.vstack((c[3],c[0])))
print(len(c[0]))
d = np.array(c)
C = tf.placeholder(tf.float32, [None, None, 1], 's')
print(d)
step = tf.reduce_mean(C)

replay_buffer = ReplayBuffer(replay_buffer_size)
al = ddpg_torch.DDPG(replay_buffer, 5,1,[16,8])
for i in range(100):
    replay_buffer.push(c[0],c[1],c[2],c[3],0)
d = 8
q_loss, policy_loss = al.update(4)
temp = np.random.random((100,5))
sort = np.random.random(10000)
print(np.array([2]))
start = time.time()
action = al.policy_net.select_action(temp,0)
quickSort(np.squeeze(action),0,99)
e1 = time.time()
print(e1-start)
#print(action)
# s1 = time.time()
# for i in range(101):
#     action = al.policy_net.select_action(c[0][1],0)
# e2=time.time()
#
# print(e2-s1)

s4=time.time()
#print(sort)
sr = quickSort(sort,0,99)
e3=time.time()
#print(sr)
print(e3-s4)
s5=time.time()
np.sort(sort)
e5 = time.time()
print(e5-s5)
#s=np.array([[[1,2,3,4,5],[2,3,4,5,6],[3,2,2,2,2]],[[2,3,4,3,4],[3,42,5,2,1]]])
#print(s.shape)
#t = al.sess.run(al.q, {al.S: s})
#print(t)
# for i in range(102):
#     al.store_transition([[1,2,3,2,3],[3,4,5,3,2]],[[1],[2]],[2],[[25,5,5,3,4],[1,2,3,2,3]])
# al.store_transition([[2, 0, 0, 0, 0], [0, 0, 0, 0, 0],[2,3,4,3,2]], [[0], [0],[1]], [[0], [0],[0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],[3,3,2,1,1]])
# print(al.memory[2][1])
# print(al.learn())

#ere = al.choose_action(np.array([2,2,3,4,5]))
#print(ere)
c = []

for i in range(1000):
    task1 = Task()
    c.append(task1.execute_time)
print(np.mean(c))



