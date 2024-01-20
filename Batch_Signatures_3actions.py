#!/usr/bin/env python
# coding=utf-8
'''
@Author: JK211
@Email: jerryren2884@gmail.com
@Date: 2023-11-27 11:59
@LastEditor: JK211
LastEditTime: 2024-1-20
@Discription:  我们基于GYMe model batch requests using a sequ编写了批量识别的环境，其中动作包括II,BSI,MRI
              We wrote the environment for batch identification based on GYM, where the actions include II,BSI,MRI
@Environment: python 3.7
'''
import gym
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
import numpy as np
import random as rand


class BatchEnvironment(gym.Env):
    def __init__(self, argv):
        """
        初始化批量接入请求环境，实际就是一批待验证的签名
        :param argv: ***参数
        """
        self.argv = argv
        self.batchSig_num = argv.batchSig_num
        self.invalid_Sig_num = argv.invalid_Sig_num

        self.Pt = 0
        self.Nt = []

        self.observation_space = Dict(
            {
                "Invalid_Percent_Pt": Box(low=0, high=100, shape=(1,), dtype=np.float32),  # 当前轮次的无效签名数量占比
                "Sigs_2be_Veri_Size": Discrete(self.batchSig_num),  # 待验证签名集合大小
                # "Sigs_2be_Veri_Nt": MultiBinary(self.batchSig_num),  # 待验证签名集合
                # "Nums_of_Detections_vt": Discrete(self.batchSig_num),  # 当前轮次检测次数
            }
        )
        # 动作空间0 1 2，其中0代表依次查询，1代表二分，2代表多分
        self.action_space = Discrete(3)

    def step(self, action=None, is_reset=True):
        """
        执行动作
        :param episode:
        :param action: 要执行的动作
        :return: 返回当前状态、奖励以及一个指示当前episode是否结束的布尔值done
        """
        ######################################################################
        #                                                                    #
        #                      init the batch environment                    #
        #                                                                    #
        ######################################################################
        Ct = []  # 用来存储验证后发现的非法签名的位置
        Mt = []  # 用来存储验证后发现的合法签名的位置
        new_Nt = []
        vt = 0   # 用来记录本轮验证所需要的验证次数
        done = False

        ######################################################################
        #                                                                    #
        #            execute action and make a state transition              #
        #                                                                    #
        ######################################################################
        if action == 0:
            # II 依次处理
            # 需要更新Ct Mt vt Nt
            for i in range(len(self.Nt)):
                vt = vt + 1
                if self.Nt[i] == 1:
                    Ct.append(i)
                else:
                    Mt.append(i)
            new_Nt = []  # 更新一下Nt
            self.lt = self.lt + vt
            uav_reward = self.get_reward(Mt, Ct, vt, new_Nt)
            uav_observation = self.get_observation(new_Nt, vt)

        elif action == 1:
            # BSI 二分处理
            # 这里先按输入进来先做一组二分
            low = 0
            high = len(self.Nt)
            mid = int((low+high)/2)
            Left_Nt = self.Nt[low:mid]
            Rigt_Nt = self.Nt[mid:high]
            if np.sum(Left_Nt) == 0:
                for i in range(len(Left_Nt)):
                    Mt.append(i)
            else:
                new_Nt = np.append(new_Nt, Left_Nt)

            if np.sum(Rigt_Nt) == 0:
                for i in range(len(Rigt_Nt)):
                    Mt.append(mid+i)  # 加mid还原为原始序列中的位置索引
            else:
                new_Nt = np.append(new_Nt, Rigt_Nt)
            vt = vt + 2
            self.lt = self.lt + vt
            uav_reward = self.get_reward(Mt, Ct, vt, new_Nt)
            uav_observation = self.get_observation(new_Nt, vt)

        else:
            # MRI 多分组处理
            # 第一步：有几个非法签名，拆分Nt为几组
            # 第二步：依次对每组进行批量验证，并记录合法签名的位置
            group_size = int(len(self.Nt)*self.Pt)
            Split_Group = np.array_split(self.Nt, group_size)
            for i in range(len(Split_Group)):
                vt = vt + 1
                if np.sum(Split_Group[i]) == 0:
                    for j in range(len(Split_Group[i])):
                        Mt.append(i*group_size + j)
                else:
                    new_Nt = np.append(new_Nt, Split_Group[i])
            self.lt = self.lt + vt
            uav_reward = self.get_reward(Mt, Ct, vt, new_Nt)
            uav_observation = self.get_observation(new_Nt, vt)

        # print("待累加的vt值为：", vt)

        if np.sum(self.Nt) == 0:
            done = True

        return uav_observation, uav_reward, done

    def get_observation(self, new_N_t, v_t):
        """
        更新观察空间，P_t
        :param N_t:
        :param e_t:
        :return: 当前投毒占比P_t,待验证签名集合N_t,当前轮次检测次数e_t
        """
        self.Pt = np.sum(new_N_t) / len(new_N_t)
        if len(new_N_t) == 0:
            self.Pt = 0
        self.Nt = new_N_t
        self.vt = v_t

        obs = np.append(self.Pt, len(self.Nt))

        return obs

    def get_reward(self, M_t, C_t, v_t, new_Nt):
        reward = len(M_t) * 1 + len(C_t) * 1 - v_t * 1

        if np.sum(new_Nt) == 0:
            reward = reward + (1/self.lt) * 1000
            # print("最后一轮的终极奖励值为：", (1/self.lt)*1000)

        return reward

    def build_environment(self, batch_num, invalid_Sig_num):
        # 初始化一个batch_num长度的01列表，用于代表发起的批量请求，其中0代表合法请求，1代表非法请求
        # invalid_Sig_num代表这里面总共有几个非法签名，即有几个1
        obs = []

        Invalid_Percent_Pt = invalid_Sig_num / batch_num
        Sigs_2be_Vefi_Nt = np.zeros(batch_num, int)
        # Nums_of_Detections_vt = 0

        for i in range(invalid_Sig_num):
            positive_position = rand.randrange(batch_num)

            while Sigs_2be_Vefi_Nt[positive_position] == 1:
                positive_position = rand.randrange(batch_num)

            Sigs_2be_Vefi_Nt[positive_position] = 1

        self.Pt = Invalid_Percent_Pt
        self.Nt = Sigs_2be_Vefi_Nt

        obs = np.append(self.Pt, len(self.Nt))

        return obs


     # 这里强制把输入固定，即非法签名占比及分布固定
    # def build_environment(self, batch_num, invalid_Sig_num):
    #
    #     self.Pt = 0.05
    #     self.Nt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #
    #     # self.Nt = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     # self.vt = Nums_of_Detections_vt
    #
    #     obs = np.append(self.Pt, len(self.Nt))
    #
    #     return obs

        # return Invalid_Percent_Pt, Sigs_2be_Vefi_Nt, Nums_of_Detections_vt

    def reset(self):
        """
        重置环境,重新生成一个随机的长度为指定的01列表，作为待验证批量请求
        :return:
        """
        self.lt = 0  # 这里用来记录算法结束后，验证出所有非法前面所需验证次数
        obs = self.build_environment(self.batchSig_num, self.invalid_Sig_num)

        return obs

    def render(self, mode="human"):
        pass

    def getN_t(self):
        return self.Nt

    def getVeri_Sum_nums(self):
        return self.lt

    def set_Nt(self):
        # 1000个签名，10个非法
        # self.Nt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # 280个签名 6个非法
        self.Nt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
