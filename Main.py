#!/usr/bin/env python
# coding=utf-8
'''
@Author: JK211
@Email: jerryren2884@gmail.com
@Date: 2023-09-19
@LastEditor: JK211
LastEditTime: 2023-09-19
@Discription: This .py is the batch identification main part for RL learning train and test
@Environment: python 3.7
'''
import os
import sys

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

import torch
# import torch_directml
import numpy as np
import argparse
from common.utils import save_results_1, make_dir
from common.utils import plot_rewards, save_args
from DQN import DQN
# from Batch_Signatures_4actions import BatchEnvironment
from Batch_Signatures_5actions import BatchEnvironment

# 保存原始的sys.stdout
original_stdout = sys.stdout

# 指定输出文件路径
output_file_path = "output.txt"

np.set_printoptions(suppress=True)

veri_nums = []


def get_args():
    """ Hyperparameters
    """
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name', default='DQN', type=str, help="name of algorithm")
    parser.add_argument('--train_eps', default=3500, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=10, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.98, type=float, help="discounted factor")
    parser.add_argument('--epsilon_start', default=0.99, type=float, help="initial value of epsilon")
    parser.add_argument('--epsilon_end', default=0.01, type=float, help="final value of epsilon")
    parser.add_argument('--epsilon_decay', default=2000, type=int, help="decay rate of epsilon")  # 值越大，保持探索的训练越长
    parser.add_argument('--lr', default=0.00020, type=float, help="learning rate")  # 0.00018
    parser.add_argument('--memory_capacity', default=100000, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--target_update', default=4, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)  # 256 300-x
    parser.add_argument('--batchSig_num', default=2000, type=int)
    parser.add_argument('--invalid_Sig_num', default=20, type=int)
    parser.add_argument('--result_path', default=curr_path + "/outputs/results/")
    parser.add_argument('--model_path', default=curr_path + "/outputs/models/")
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()
    # args.device = torch_directml.device(0)
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU
    args.device = torch.device("cpu")  # check GPU

    return args


def env_agent_config(cfg, seed=1):
    """
    创建环境和智能体
    """
    env = BatchEnvironment(argv=cfg)  # 创建环境
    # n_states = len(env.observation_space.spaces)  # 状态维度
    n_states = 2  # 状态维度
    n_actions = env.action_space.n  # 动作维度
    print(f"n states: {n_states}, n actions: {n_actions}")
    agent = DQN(n_states, n_actions, cfg)  # 创建智能体
    if seed != 0:  # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
    return env, agent


def train(cfg, env, agent):
    ''' Training
    '''
    print('Start training!')
    print(f' 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    steps = []
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        last_state = state
        last_action = 10
        while True:
            # print("第{}次训练的{}步的状态为{}".format(i_ep+1, ep_step+1, state))
            # print("执行动作前待验证集合情况：", env.getN_t())
            action = agent.choose_action(state, last_state, last_action)  # 选择动作
            last_action = action
            last_state = state
            # print("++++++++++", action)
            next_state, reward, done = env.step(action)  # 更新环境，返回transition
            print("第{}次训练的{}步的动作值为{}".format(i_ep + 1, ep_step + 1, action))
            # print("第{}次训练的{}步的done值为{}".format(i_ep, ep_step, done))
            # print("执行动作后待验证集合情况：", env.getN_t())
            # print("执行动作后待验证集合长度：", len(env.getN_t()))
            print("第{}次训练的{}步动作后待验证集合长度为{}：".format(i_ep + 1, ep_step + 1, len(env.getN_t())))
            # print("+++++++++++++++++++")
            agent.memory.push(state, action, reward, next_state, done)  # 保存transition
            state = next_state  # 更新下一个状态
            agent.update()  # 更新智能体
            ep_reward += reward  # 累加奖励
            ep_step += 1
            if done:
                # print("结束时待验证集合的情况为：", env.getN_t())
                print("第{}次结束时总的验证次数为：{}".format(i_ep + 1, env.getVeri_Sum_nums()))
                veri_nums.append(env.getVeri_Sum_nums())
                break

        if (i_ep + 1) % cfg.target_update == 0:  # 智能体目标网络更新
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        steps.append(ep_step)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 1 == 0:
            print(
                f'Episode：{i_ep + 1}/{cfg.train_eps}, Reward:{ep_reward:.2f}, Step:{ep_step:.2f} Epislon:{agent.epsilon(agent.frame_idx):.3f}')
            print("--------------------------------------------------------------------------")
    print('Finish training!')
    env.close()
    res_dic = {'rewards': rewards, 'ma_rewards': ma_rewards, 'steps': steps}
    return res_dic


def test(cfg, env, agent):
    print('开始测试!')
    print(f'算法：{cfg.algo_name}, 设备：{cfg.device}')
    ############# 由于测试不需要使用epsilon-greedy策略，所以相应的值设置为0 ###############
    cfg.epsilon_start = 0.0  # e-greedy策略中初始epsilon
    cfg.epsilon_end = 0.0  # e-greedy策略中的终止epsilon
    ################################################################################
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    steps = []
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        env.set_Nt()
        last_state = state
        last_action = 10
        while True:
            # print("第{}次测试的{}步的状态为{}".format(i_ep + 1, ep_step + 1, state))

            action = agent.choose_action(state, last_state, last_action)  # 选择动作
            print("第{}次测试的{}步的动作值为{}".format(i_ep + 1, ep_step + 1, action))
            last_action = action
            last_state = state
            # print("++++++++++", action)
            next_state, reward, done = env.step(action)  # 更新环境，返回transition
            # print("执行动作后待验证集合情况：", env.getN_t())
            # print("执行动作后待验证集合长度：", len(env.getN_t()))
            print("第{}次测试的{}步动作后待验证集合长度为{}：".format(i_ep + 1, ep_step + 1, len(env.getN_t())))
            # print("+++++++++++++++++++++++++++++++++++++++")
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            ep_step += 1
            if done:
                print("测试第{}次结束时总的验证次数为：{}".format(i_ep + 1, env.getVeri_Sum_nums()))
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f'Episode：{i_ep + 1}/{cfg.test_eps}, Reward:{ep_reward:.2f}, Step:{ep_step:.2f}')
        print("-----------------------------------------------------------------------------------")
    print('完成测试！')
    env.close()
    return {'rewards': rewards, 'ma_rewards': ma_rewards, 'steps': steps}


if __name__ == "__main__":
    try:
        # 打开文件以写入模式，并将sys.stdout重定向到文件
        # with open(output_file_path, 'a', encoding='utf-8') as file:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            sys.stdout = file

            print("+" + '-' * 100 + "+")
            print("+" + '-' * 100 + "+")
            print("+" + '-' * 100 + "+")

            # 你的现有代码和输出语句在这里
            cfg = get_args()
            # 训练
            env, agent = env_agent_config(cfg)
            if os.path.exists(cfg.model_path + "/dqn_checkpoint.pth"):
                agent.load(cfg.model_path)
            res_dic = train(cfg, env, agent)
            make_dir(cfg.result_path, cfg.model_path)  # 创建保存结果和模型路径的文件夹
            save_args(cfg)
            agent.save(path=cfg.model_path)  # 保存模型
            list.sort(veri_nums)
            print("训练阶段探索到的最低验证次数为：", veri_nums[0])
            print("训练验证次数一览：", veri_nums)
            save_results_1(res_dic, tag='train', path=cfg.result_path)  # 保存结果
            plot_rewards(res_dic['rewards'], res_dic['ma_rewards'], cfg, tag="train")  # 画出结果
            # 测试
            env, agent = env_agent_config(cfg)
            agent.load(path=cfg.model_path)  # 导入模型
            print("模型地址为：", cfg.model_path)
            res_dic = test(cfg, env, agent)
            save_results_1(res_dic, tag='test', path=cfg.result_path)  # 保存结果
            plot_rewards(res_dic['rewards'], res_dic['ma_rewards'], cfg, tag="test")  # 画出结果

    finally:
        # 恢复原始的sys.stdout
        sys.stdout = original_stdout
