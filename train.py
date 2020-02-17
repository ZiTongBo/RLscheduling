#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/8

from ddpg import *
from env import *
from EDF import *
if __name__ == '__main__':
    replay_buffer = ReplayBuffer(replay_buffer_size)
    np.set_printoptions(precision=2, suppress=True)
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
            if len(env.instance) > 0:
                actions = GlobalEDF(env.instance, env.no_processor)
                reward, _, info = env.step(actions)
                edf_completed += info[0]
                edf_episode_reward = edf_completed
                if env.done():
                    break
        env.load()
        for step in range(max_steps):
            actions = []
            states = []
            env.arrive()
            no_instance = len(env.instance)
            for i in env.instance:
                action = [0]
                state = env.observation(i)
                if i_episode > explore_episodes and i_episode % 5 != 0:
                    action = alg.policy_net.select_action(state, noise)
                    noise *= 0.999686
                else:
                    action = alg.policy_net.sample_action(action_range=1)
                actions.append(np.array(action[0]))
                print('state',state)
                states.append(state)
            if DEBUG:
                print(actions)
            reward, done, next_state, info = env.step(actions)
            completed += info[0]
            missed += info[1]
            for i in range(no_instance):
                replay_buffer.push(states[i], [actions[i]], reward[i], next_state[i], done[i])
                if DEBUG:
                    print('time:', env.time, 'noise:', noise)
                    #print('state:', states[i], 'action:', actions[i], 'reward:', reward[i], 'next_state:',
                    #      next_state[i], 'done:', done[i])
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

