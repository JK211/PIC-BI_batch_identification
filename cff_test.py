#!/usr/bin/env python
# coding=utf-8
'''
@Author: JK211
@Email: jerryren2884@gmail.com
@Date: 2023-11-27 11:59
@LastEditor: JK211
LastEditTime: 2024-1-20
@Discription:  这是利用cff做批量识别的单独实现
              This is a separate implementation of doing batch recognition using cff
@Environment: python 3.7
'''
import numpy as np
import math
np.set_printoptions(suppress=True)

Nt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

lt = 0
vt = 0
Mt = []  # 识别为合法的签名
Ct = []  # 识别为非法的签名

new_Nt = []
Split_Group = []
n = 125
m = 25
d = 2

# n = 27
# m = 9
# d = 1
#  先将Nt进行分组，组大小为125
group_size = n - 1
group_nums = len(Nt) / n
ceil_group_nums = math.ceil(group_nums)  # 向上取整
for i in range(ceil_group_nums):
    if i < int(group_nums):
        temp = Nt[(group_size * i + i): (group_size * i + i) + group_size + 1]
        Split_Group.append(temp)
    else:
        temp = Nt[(group_size * i + i): (len(Nt) - 1) + 1]
        Split_Group.append(temp)
if len(Split_Group[ceil_group_nums - 1]) > n / 2:
    # 如果最后一组数量少于n/2，将其补齐为n
    temp = [0] * (n - len(Split_Group[ceil_group_nums - 1]))
    Split_Group[ceil_group_nums - 1] = np.append(Split_Group[ceil_group_nums - 1], temp)
#  分组后，依次对每组执行2-CFF(25, 125)，如果一组没被补齐，那就跳过并加入Nt下一轮再处理
index = 0
# M = np.load("./Cover free families demos/1-cff9-27.npy")
M = np.load("./Cover free families demos/2-cff25-125.npy")
print("Split_Group大小为：", len(Split_Group))
print("Split Group为：", Split_Group)

# while len(Split_Group[index]) == n:

all_dist_keys = []
for item in Split_Group:
    if len(item) == n:
        # print("+++++++++++++++")
        print("item为：", item)
        valid_set = []
        invalid_set = []
        indist_set = []
        count = 0
        cff_batch = []
        new_Nt_keys = []
        vt = vt + m
        for i in range(len(M)):
            # 这里按25进行分组，每组125，并打上序号，方便后续处理
            temp = {}
            for j in range(len(M[i])):
                if M[i][j] == 1:
                    temp[j] = item[j]
            cff_batch.append(temp)
        # print("-------------", cff_batch)

        all_keys = [x for x in range(0, len(M[0]), 1)]
        count = np.sum(item)
        for subset in cff_batch:
            subset_keys = list(subset.keys())
            subset_values = list(subset.values())

            if np.sum(subset_values) == 0:
                valid_set = list(set(valid_set).union(subset_keys))
        Mt = np.append(Mt, valid_set)

        if count <= d:
            invalid_set = list(set(all_keys).difference(valid_set))
            Ct = np.append(Ct, invalid_set)
        else:
            indist_set = list(set(all_keys).difference(valid_set))
            all_dist_keys = np.append(all_dist_keys, indist_set)
            new_Nt_keys = np.append(new_Nt_keys, indist_set)
            # indist_set_values = [x for x in subset[indist_set]]

            indist_set_values = []
            for x in indist_set:
                indist_set_values.append(item[x])
            new_Nt = np.append(new_Nt, indist_set_values)

if len(Split_Group[len(Split_Group) - 1]) < n:
    new_Nt = np.append(new_Nt, Split_Group[len(Split_Group) - 1])
lt = lt + vt

print("合法签名集合Mt为：", Mt)
print("合法签名数量为：", len(Mt))
print("非法签名集合Ct为：", Ct)
print("非法签名数量为：", len(Ct))
print("new_Nt_keys为：", new_Nt_keys)
print("new_Nt为：", list(new_Nt))
print(len(new_Nt))
print("all_dist_keys为：", all_dist_keys)