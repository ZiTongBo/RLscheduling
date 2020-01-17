#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/8

from ddpg import *
from env import *
from EDF import *
if __name__ == '__main__':
    replay_buffer = ReplayBuffer(replay_buffer_size)
    np.set_printoptions(precision=6, suppress=True)
    alg = DDPG(replay_buffer, state_dim, action_dim, hidden_dim)
    # alg.load_model(model_path)

    # hyper-parameters
    noise = 3
    frame_idx = 0
    mean_rewards = []
    mean_edf_rewards = []
    rewards = []
    env = Env()
    edf_rewards = []
    for i_episode in range(max_episodes):
        q_loss_list = []
        policy_loss_list = []
        env.reset()
        episode_reward = 0
        edf_episode_reward = 0
        completed = 0
        edf_completed = 0
        missed = 0
        env.save()
        for step in range(max_steps):
            env.arrive()
            actions = GlobalEDF(env.taskSet, env.noProcessor)
            reward, _, info = env.step(actions)
            edf_completed += info[0]
            for i in range(env.noTask):
                if actions[i] != 0:
                    #edf_episode_reward += reward[i]
                    edf_episode_reward = info[0]
            edf_episode_reward = edf_completed
            if env.done():
                break
        env.load()
        for step in range(max_steps):
            actions = []
            states = []
            env.arrive()
            for task in env.taskSet:
                action = [0]
                state = [0]
                if task.isArrive:
                    state = env.observation(task)
                    if i_episode > explore_episodes and i_episode % 5 != 0:
                        action = alg.policy_net.select_action(state, noise)
                        noise *= 0.999686
                    else:
                        action = alg.policy_net.sample_action(action_range=1)
                actions.append(np.array(action[0]))
                states.append(state)

            reward, done, info = env.step(actions)
            completed += info[0]
            missed += info[1]
            for i in range(env.noTask):
                if actions[i] != 0:
                    replay_buffer.push(states[i], [actions[i]], reward[i], env.observation(env.taskSet[i]), done[i])
                    if i_episode % 101 == 0 and i_episode > 200:
                        print('time:', env.time, 'noise:', noise)
                        print(states[i],np.array([actions[i]]), [reward[i]], env.observation(env.taskSet[i]), [done[i]])
                    #episode_reward += reward[i]

            episode_reward = completed
            frame_idx += 1

            if len(replay_buffer) > batch_size and step % 5 == 0:
                q_loss, policy_loss = alg.update(batch_size)
                q_loss_list.append(q_loss)
                policy_loss_list.append(policy_loss)
            if env.done():
                break
        if i_episode % 5 != 0:
            mean_rewards.append(episode_reward)
        mean_edf_rewards.append(edf_episode_reward)
        if i_episode % 5 == 0 and i_episode > explore_episodes:
            rewards.append(np.mean(mean_rewards))
            edf_rewards.append(np.mean(mean_edf_rewards))
            mean_edf_rewards = []
            mean_rewards = []
            if i_episode % 40 == 0:
                plot(rewards, edf_rewards)
                alg.save_model(model_path)
        print('Eps: ', i_episode, '| Reward: ', int(episode_reward), '| Loss: %.2f' % np.average(q_loss_list),'%.2f' %
              np.average(policy_loss_list), '| Completed & Missed:', int(completed), int(missed))

