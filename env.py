#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/9

from task import *


# TODO: 任务到达，终止条件
class Env(object):
    def __init__(self):
        self.time = 0
        self.taskSet = []
        self.noProcessor = 0
        self.noTask = 0
        self.meanDeadline = 0
        self.meanExecuteTime = 0
        self.minDeadline = 999
        self.minExecuteTime = 999

    def reset(self):
        self.time = 0
        self.noProcessor = 2
        self.noTask = 4
        self.meanDeadline = 0
        self.meanExecuteTime = 0
        self.minDeadline = 999
        self.minExecuteTime = 999
        self.taskSet = []
        for i in range(self.noTask):
            task = Task()
            self.taskSet.append(task)
        self.arrive()
        self.update()

    def step(self, actions):
        reward = 0
        exec_task = np.argsort(actions)[::-1]
        for i in range(self.noProcessor):
            if self.taskSet[exec_task[i]].isArrive:
                if self.taskSet[exec_task[i]].execute():
                    reward += 1
                    print("finish")
        for i in range(self.noTask):
            if self.taskSet[i].isArrive:
                self.taskSet[i].deadline -= 1
                if self.taskSet[i].deadline == 0:
                    self.taskSet[i].miss()
                    reward -= 1
                    print("miss")
        self.time += 1
        self.arrive()
        self.update()
        return reward, self.done()

    def update(self):
        total_deadline = 0
        total_execute_time = 0
        for i in range(self.noTask):
            task = self.taskSet[i]
            total_deadline += task.deadline
            total_execute_time += task.executeTime
            if task.deadline < self.minDeadline:
                self.minDeadline = task.deadline
            if task.executeTime < self.minExecuteTime:
                self.minExecuteTime = task.executeTime
        self.meanDeadline = total_deadline / self.noTask
        self.meanExecuteTime = total_execute_time / self.noTask

    def arrive(self):
        for task in self.taskSet:
            if not task.isArrive and self.time % task.period == 0:
                task.arrive()
        pass

    def done(self):
        for i in range(self.noTask):
            if self.taskSet[i].count > 0:
                return False
        return True

    def observation(self, task):
        return np.array([task.executeTime, task.deadline, task.period,
                         self.noTask, self.meanDeadline, self.minDeadline, self.meanExecuteTime, self.minExecuteTime])

