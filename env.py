#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/9

from task import *
import copy as cp


# TODO: 任务到达，终止条件
class Env(object):
    def __init__(self):
        self.time = 0
        self.taskSet = []
        self.noProcessor = 0
        self.noTask = 0
        self.meanDeadline = 0
        self.meanExecuteTime = 0
        self.minDeadline = np.inf
        self.arrTask = 0
        self.minExecuteTime = np.inf
        self.saved = []

    def reset(self):
        self.time = 0
        self.noProcessor = 5
        self.noTask = 10
        self.arrTask = 0
        self.meanDeadline = 0
        self.meanExecuteTime = 0
        self.minDeadline = 999
        self.minExecuteTime = 999
        self.taskSet = []
        for i in range(self.noTask):
            task = Task()
            self.taskSet.append(task)
        self.update()

    def step(self, actions):
        reward = np.zeros(self.noTask)
        global_reward = 0
        done = np.zeros(self.noTask)
        exec_task = np.argsort(actions)[::-1]
        info = np.zeros(2)
        for i in range(self.noProcessor):
            if self.taskSet[exec_task[i]].isArrive and actions[exec_task[i]]>0:
                self.taskSet[exec_task[i]].execute()
                if not self.taskSet[exec_task[i]].isArrive:
                    reward[exec_task[i]] += 1
                    info[0] += 1
                    global_reward += 1
                    self.taskSet[exec_task[i]].reDeadline -= 1
                    self.arrTask -= 1
                    done[exec_task[i]] = 1
        for i in range(self.noTask):
            self.taskSet[i].time += 1
            if self.taskSet[i].isArrive:
                self.taskSet[i].reDeadline -= 1
                if self.taskSet[i].reDeadline == 0:
                    self.taskSet[i].miss()
                    reward[i] -= 1
                    global_reward -= 1
                    self.arrTask -= 1
                    done[i] = 1
                    info[1] += 1
        self.time += 1
        self.update()
        return reward + global_reward, done, info

    def update(self):
        total_deadline = 0
        arr_task_deadline = np.zeros(self.arrTask)
        arr_task_execute = np.zeros(self.arrTask)
        total_execute_time = 0
        self.minDeadline = 999
        self.minExecuteTime = 999
        if self.arrTask == 0:
            self.meanDeadline = 0
            self.meanExecuteTime = 0
            self.minDeadline = 0
            self.minExecuteTime = 0
            return
        for i in range(self.noTask):
            task = self.taskSet[i]
            if task.isArrive:
                total_deadline += task.reDeadline
                total_execute_time += task.reExecuteTime
                if task.reDeadline < self.minDeadline:
                    self.minDeadline = task.reDeadline
                if task.reExecuteTime < self.minExecuteTime:
                    self.minExecuteTime = task.reExecuteTime
        self.meanDeadline = total_deadline / self.arrTask
        self.meanExecuteTime = total_execute_time / self.arrTask

    def arrive(self):
        for task in self.taskSet:
            if not task.isArrive and task.time % task.period == 0 and task.count > 0:
                task.arrive()
                self.arrTask += 1
        self.update()

    def done(self):
        if self.arrTask > 0:
            return False
        for i in range(self.noTask):
            if self.taskSet[i].count > 0:
                return False
        return True

    def observation(self, task):
        return np.array([task.reExecuteTime, task.reDeadline, task.period,
                         self.arrTask, self.meanExecuteTime, self.minExecuteTime , self.meanDeadline, self.minDeadline])

    def save(self):
        self.saved = cp.deepcopy(self.taskSet)

    def load(self):
        self.time = 0
        self.noProcessor = 5
        self.noTask = 10
        self.arrTask = 0
        self.meanDeadline = 0
        self.meanExecuteTime = 0
        self.minDeadline = 999
        self.minExecuteTime = 999
        self.taskSet = self.saved
        self.update()
