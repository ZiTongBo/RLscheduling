#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/8

from ddpg_torch import *
from env import *
from EDF import *
if __name__ == '__main__':
    replay_buffer = ReplayBuffer(replay_buffer_size)
    np.set_printoptions(precision=2, suppress=True)
    alg = DDPG(replay_buffer, state_dim, action_dim, HIDDEN)
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
                if env.done():
                    break
            env.load()
        for step in range(max_steps):
            # env.arrive()
            no_instance = len(env.instance)
            actions = np.ones((no_instance, 1), dtype=np.float32)
            states = np.ones((no_instance, state_dim), dtype=np.float32)
            if no_instance > env.no_processor:
                for i in range(no_instance):
                    state = env.observation(env.instance[i])
                    if i_episode > explore_episodes:
                        action = alg.policy_net.select_action(state, noise)
                    else:
                        action = alg.policy_net.sample_action(action_range=1)
                    actions[i] = action
                    states[i] = state
                noise *= 0.9686

            reward, done, next_state, info = env.step(np.squeeze(actions))
            # info[0]: 执行成功数 info[1]：执行失败数
            completed += info[0]
            missed += info[1]
            if no_instance > env.no_processor:
                replay_buffer.push(states, actions, reward, next_state, done)
            episode_reward = completed
            frame_idx += 1

            if len(replay_buffer) > BATCH_SIZE and step % 200 == 0:
                q_loss, policy_loss = alg.update(BATCH_SIZE)
                alg.update(BATCH_SIZE)
                alg.update(BATCH_SIZE)
                alg.update(BATCH_SIZE)
                alg.update(BATCH_SIZE)
                q_loss_list.append(q_loss)
                policy_loss_list.append(policy_loss)
            if env.done():
                break
        mean_rewards.append(episode_reward)
        mean_edf_rewards.append(edf_episode_reward)
        if i_episode % 1 == 0 and i_episode >= explore_episodes:
            rewards.append(np.mean(mean_rewards)*100/(env.no_task*FREQUENCY))
            edf_rewards.append(np.mean(mean_edf_rewards)*100/(env.no_task*FREQUENCY))
            mean_edf_rewards = []
            mean_rewards = []
            if i_episode % 1 == 0:
                alg.plot(rewards, edf_rewards)
                alg.save_model(model_path)
        print('Eps: ', i_episode, ' | Step: ', step, '| Reward: ', int(episode_reward), ' | Loss: %.2f' % np.average(q_loss_list),'%.2f' %
              np.average(policy_loss_list), '| Completed & Missed:', int(completed), int(missed))