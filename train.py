#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/8

from ddpg import *
from env import *
if __name__ == '__main__':
    replay_buffer = ReplayBuffer(replay_buffer_size)

    alg = DDPG(replay_buffer, state_dim, action_dim, hidden_dim)
    # alg.load_model(model_path)

    # hyper-parameters

    frame_idx = 0
    rewards = []
    env = Env()
    for i_episode in range(max_episodes):
        q_loss_list = []
        policy_loss_list = []

        env.reset()
        episode_reward = 0

        for step in range(max_steps):
            actions = []
            states = []
            artask = 0
            for task in env.taskSet:
                action = [0]
                state = [0]
                print(task.deadline,task.isArrive,task.executeTime)
                if task.isArrive:
                    artask += 1
                    state = env.observation(task)
                    if frame_idx > explore_steps:
                        action = alg.policy_net.select_action(state)
                    else:
                        action = alg.policy_net.sample_action(action_range=1.)
                actions.append(action[0])
                states.append(state)
                #print(action)
            print("time",env.time, "artask", artask)
            reward, done = env.step(actions)
            print("reward", reward)
            for i in range(env.noTask):
                if actions[i] != 0:
                    replay_buffer.push(states[i], np.array([actions[i]]), reward, env.observation(env.taskSet[i]), done)
                    print(states[i], np.array([actions[i]]), [reward], env.observation(env.taskSet[i]), [done])
            episode_reward += reward
            frame_idx += 1
            if len(replay_buffer) > batch_size:
                q_loss, policy_loss = alg.update(batch_size)
                q_loss_list.append(q_loss)
                policy_loss_list.append(policy_loss)
            if done:
                break
        if i_episode % 20 == 0:
            plot(rewards)
            alg.save_model(model_path)
        print('Eps: ', i_episode, '| Reward: ', episode_reward, '| Loss: ', np.average(q_loss_list),
              np.average(policy_loss_list))

        rewards.append(episode_reward)