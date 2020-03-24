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
import datetime
from env import *
from timeit import timeit
from timeit import repeat

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
t = []
e = 0
with tf.variable_scope('foo'):
    v = tf.get_variable('v', [1], initializer=tf.constant_initializer(1.0))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return ()


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


# 返回 x 在 arr 中的索引，如果不存在返回 -1
def binarySearch(arr, l, r, x):
    # 基本判断
    if r >= l:

        mid = int(l + (r - l) / 2)

        # 元素整好的中间位置
        if arr[mid] == x:
            return mid

            # 元素小于中间位置的元素，只需要再比较左边的元素
        elif arr[mid] > x:
            return binarySearch(arr, l, mid - 1, x)

            # 元素大于中间位置的元素，只需要再比较右边的元素
        else:
            return binarySearch(arr, mid + 1, r, x)

    else:
        # 不存在
        return -1


# 测试数组
arr = np.random.random(10000000)
print(arr)
np.sort(arr)
x = 0.323
st = time.time()

# 函数调用
result = binarySearch(arr, 0, len(arr) - 1, x)
ts = time.time()
print(ts - st)
replay = ReplayBuffer(1000)
ddpg = ddpg_torch.DDPG(replay, 5, 1, [32, 16])

np.set_printoptions(precision=2, suppress=True)
input = 20
result = []
w1 = np.random.random((5, 32))
b1 = np.random.random((input, 32))
w2 = np.random.random((32, 16))
b2 = np.random.random((input, 16))
w3 = np.random.random((16, 1))
b3 = np.random.random((input, 1))
for i in range(10):
    t1 = time.process_time()
    for j in range(10):
        temp = np.random.random((input, 5))
        o1 = sigmoid(np.add(np.matmul(temp, w1), b1))
        o2 = sigmoid(np.add(np.matmul(o1, w2), b2))
        o3 = np.add(np.matmul(o2, w3), b3)
        np.sort(np.squeeze(o3), kind='quicksort')
    t3 = time.process_time()
    result.append(t3 - t1)

print('20:', min(result) * 100, np.mean(result) * 100, max(result) * 100)
input = 50
result = []
w1 = np.random.random((5, 32))
b1 = np.random.random((input, 32))
w2 = np.random.random((32, 16))
b2 = np.random.random((input, 16))
w3 = np.random.random((16, 1))
b3 = np.random.random((input, 1))
for i in range(100):
    t1 = time.process_time()
    for j in range(10):
        temp = np.random.random((input, 5))
        o1 = sigmoid(np.add(np.matmul(temp, w1), b1))
        o2 = sigmoid(np.add(np.matmul(o1, w2), b2))
        o3 = np.add(np.matmul(o2, w3), b3)
        np.sort(np.squeeze(o3), kind='quicksort')
    t3 = time.process_time()
    result.append(t3 - t1)
    # np.sort(np.squeeze(o3),kind='quicksort')
print('50:', min(result) * 100, np.mean(result) * 100, max(result) * 100)
input = 50
result = []
w1 = np.random.random((5, 32))
b1 = np.random.random((input, 32))
w2 = np.random.random((32, 16))
b2 = np.random.random((input, 16))
w3 = np.random.random((16, 1))
b3 = np.random.random((input, 1))
for i in range(100):
    t1 = time.process_time()
    for j in range(10):
        temp = np.random.random((input, 5))
        o1 = np.maximum(np.add(np.matmul(temp, w1), b1), 0)
        o2 = np.maximum(np.add(np.matmul(o1, w2), b2), 0)
        o3 = np.add(np.matmul(o2, w3), b3)
        np.sort(np.squeeze(o3), kind='quicksort')
    t3 = time.process_time()
    result.append(t3 - t1)
    # np.sort(np.squeeze(o3),kind='quicksort')
print('50:', min(result) * 100, np.mean(result) * 100, max(result) * 100)
input = 100
result = []
w1 = np.random.random((5, 32))
b1 = np.random.random((input, 32))
w2 = np.random.random((32, 16))
b2 = np.random.random((input, 16))
w3 = np.random.random((16, 1))
b3 = np.random.random((input, 1))
for i in range(100):
    t1 = time.process_time()
    for j in range(10):
        temp = np.random.random((input, 5))
        o1 = np.maximum(np.add(np.matmul(temp, w1), b1), 0)
        o2 = np.maximum(np.add(np.matmul(o1, w2), b2), 0)
        o3 = np.add(np.matmul(o2, w3), b3)
        # np.sort(np.squeeze(o3), kind='quicksort')
    t3 = time.process_time()
    result.append(t3 - t1)
    # np.sort(np.squeeze(o3),kind='quicksort')
print('100:', min(result) * 100, np.mean(result) * 100, max(result) * 100)
input = 200
result = []
liner = []
non = []
w1 = np.random.random((5, 32))
b1 = np.random.random((input, 32))
w2 = np.random.random((32, 16))
b2 = np.random.random((input, 16))
w3 = np.random.random((16, 1))
b3 = np.random.random((input, 1))
for i in range(1000):
    t1 = time.clock()
    for j in range(1):
        temp = np.random.random((input, 5))
        l1 = time.clock()
        o1 = np.add(np.matmul(temp, w1), b1)
        l2 = time.clock()
        o1 = sigmoid(o1)
        l3 = time.clock()
        o2 = np.add(np.matmul(o1, w2), b2)
        l4 = time.clock()
        o2 = sigmoid(o2)
        l5 = time.clock()
        o3 = np.add(np.matmul(o2, w3), b3)
        l6 = time.clock()
        # np.sort(np.squeeze(o3), kind='quicksort')
        liner.append(l6 - l5 + l4 - l3 + l2 - l1)
        non.append(l5 - l4 + l3 - l2)
    t3 = time.clock()
    result.append(t3 - t1)

    # np.sort(np.squeeze(o3),kind='quicksort')
# print(np.array(result) * 1000)
print(liner)
print(non)
print(np.mean(liner) / (np.mean(liner) + np.mean(non)))
print('200:', min(result) * 1000, np.mean(result) * 1000, max(result) * 1000)

result = []
input = 200
w1 = np.random.random((8, 16))
b1 = np.random.random((input, 16))
w2 = np.random.random((16, 8))
b2 = np.random.random((input, 8))
w3 = np.random.random((8, 1))
b3 = np.random.random((input, 1))

temp = np.random.random((input, 8))


def nn():
    o1 = np.maximum(np.add(np.matmul(temp, w1), b1) ,0)
    o2 = np.maximum(np.add(np.matmul(o1, w2), b2), 0)
    o3 = np.add(np.matmul(o2, w3), b3)

def nnn():
    temp = np.random.random((input, 8))


te = repeat('nnn()', 'from __main__ import nnn', number=1,repeat=10)
te = []
for i in range(10):
    temp = np.random.random((input, 8))
    t = timeit('nn()', 'from __main__ import nn', number=1)
    te.append(t)
print('new:', min(te)*1000,np.mean(te)*1000,max(te)*1000)
