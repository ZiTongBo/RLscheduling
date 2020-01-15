#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/8

from ddpg import *
from env import *
if __name__ == '__main__':
    replay_buffer = ReplayBuffer(replay_buffer_size)
    np.set_printoptions(precision=3, suppress=True)
    alg = DDPG(replay_buffer, state_dim, action_dim, hidden_dim)
    # alg.load_model(model_path)

    # hyper-parameters
    noise = 3
    frame_idx = 0
    rewards = []
    env = Env()
    for i_episode in range(max_episodes):
        q_loss_list = []
        policy_loss_list = []
        env.reset()
        episode_reward = 0
        completed = 0
        missed = 0
        for step in range(max_steps):
            actions = []
            states = []
            env.arrive()
            for task in env.taskSet:
                action = [0]
                state = [0]
                if task.isArrive:
                    state = env.observation(task)
                    if i_episode > explore_episodes:
                        action = alg.policy_net.select_action(state, noise)
                        noise *= 0.999686
                    else:
                        action = alg.policy_net.sample_action(action_range=1)
                actions.append(np.array(action[0]))
                states.append(state)
            # print('time:', env.time, 'noise:', noise)
            reward, done, info = env.step(actions)
            completed += info[0]
            missed += info[1]
            for i in range(env.noTask):
                if actions[i] != 0:
                    replay_buffer.push(states[i], [actions[i]], reward[i], env.observation(env.taskSet[i]), done[i])
                    # print(states[i],np.array([actions[i]]), [reward[i]], env.observation(env.taskSet[i]), [done[i]])
                    episode_reward += reward[i]
            frame_idx += 1

            if len(replay_buffer) > batch_size:
                q_loss, policy_loss = alg.update(batch_size)
                q_loss_list.append(q_loss)
                policy_loss_list.append(policy_loss)
            if env.done():
                break
        if i_episode % 100 == 0:
            plot(rewards)
            alg.save_model(model_path)
        print('Eps: ', i_episode, '| Reward: ', int(episode_reward), '| Loss: %.2f' % np.average(q_loss_list),'%.2f' %
              np.average(policy_loss_list), '| Completed & Missed:', int(completed), int(missed))

        rewards.append(episode_reward)