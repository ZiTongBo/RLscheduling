#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/8

from ddpg import *
from env import *
from EDF import *
if __name__ == '__main__':
    replay_buffer = ReplayBuffer(replay_buffer_size)
    np.set_printoptions(precision=5, suppress=True)
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
        total = 0
        missed = 0
        edf_missed = 0
        if EDF:
            env.save()
            for step in range(max_steps):
                # env.arrive()
                actions = GlobalEDF(env.instance, env.no_processor)
                reward, _, _, info = env.step(actions)
                edf_completed += info[0]
                edf_missed += info[1]
                edf_episode_reward = edf_completed
                if (step+1) % 1000 == 0:
                    edf_rewards.append(edf_completed * 100 / (edf_completed + edf_missed))
                if env.done():
                    break
            env.load()
        for step in range(max_steps):
            # env.arrive()
            no_instance = len(env.instance)
            actions = np.ones(no_instance)
            states = []
            if no_instance > env.no_processor:
                for i in range(no_instance):
                    state = env.observation(env.instance[i])
                    if i_episode > explore_episodes:
                        action = alg.policy_net.select_action(state, noise)
                    else:
                        action = alg.policy_net.sample_action(action_range=1)
                    actions[i] = np.array(action[0])
                    states.append(state)
                noise *= 0.999686
            reward, done, next_state, info = env.step(actions)
            # info[0]: 执行成功数 info[1]：执行失败数
            completed += info[0]
            missed += info[1]
            for i in range(no_instance):
                total += done[i]
            if no_instance > env.no_processor:
                for i in range(no_instance):
                    replay_buffer.push(states[i], [actions[i]], reward[i], next_state[i], done[i])
                    if DEBUG and step % 1000 == 0:
                        print('state:', states[i], 'action:', actions[i], 'reward:', reward[i], 'next_state:',
                              next_state[i], 'done:', done[i])
                    # episode_reward += reward[i]
            episode_reward = completed
            frame_idx += 1
            if (step+1) % 1000 == 0:
                rewards.append(completed * 100 / (completed + missed))
            if len(replay_buffer) > batch_size and step % 10 == 0:
                q_loss, policy_loss = alg.update(batch_size)
                q_loss_list.append(q_loss)
                policy_loss_list.append(policy_loss)
            if env.done():
                break
        mean_rewards.append(episode_reward)
        mean_edf_rewards.append(edf_episode_reward)
        if i_episode % 1 == 0 and i_episode >= explore_episodes:
            # rewards.append(np.mean(mean_rewards)*100/(env.no_task*FREQUENCY))
            # edf_rewards.append(np.mean(mean_edf_rewards)*100/(env.no_task*FREQUENCY))
            mean_edf_rewards = []
            mean_rewards = []
            if i_episode % 1 == 0:
                print(rewards,edf_rewards)
                alg.plot(rewards, edf_rewards)
                alg.save_model(model_path)
        print('Eps: ', i_episode, ' | Step: ', step, '| Reward: ', int(episode_reward), '| Loss: %.2f' % np.average(q_loss_list),'%.2f' %
              np.average(policy_loss_list), '| Completed & Missed:', int(completed), int(missed))

