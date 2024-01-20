#!/usr/bin/env python
# coding=utf-8
'''
@Author: JK211
@Email: jerryren2884@gmail.com
@Date: 2023-11-12 16:44
@LastEditor: JK211
LastEditTime: 2023-11-12 16:44
@Discription:  初步实现了估计算法的功能，但目前的问题是预测准确时，预测值的权重没有马上赋予比较大的值
               幸运的是，当攻击者行为变化时（即非法前面占比变化剧烈时）会触发重采样，采样值权重会逼近1，符合预期
@Environment: python 3.8
'''
import numpy as np
import pandas as pd
import math
import random as rand
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from matplotlib import pyplot as plt
#新增加的两行
import matplotlib
import csv
matplotlib.rc("font",family='YouYuan')
from sklearn.metrics import mean_squared_error
from brokenaxes import brokenaxes




def batch_request_generating(batch_size, invalid_size):
    Sigs2beVefi = np.zeros(batch_size, int)
    for i in range(invalid_size):
        positive_position = rand.randrange(batch_size)

        while Sigs2beVefi[positive_position] == 1:
            positive_position = rand.randrange(batch_size)

        Sigs2beVefi[positive_position] = 1

    return Sigs2beVefi


# 预测分析
def forecast(hisResult):

    if(len(hisResult) == 0):
        # p[0] = 0
        p = 0
    elif(len(hisResult) == 1):
        # p[1] = hisResult[0]
        p = hisResult[0]
    else:
        y3 = pd.Series(hisResult)
        # ets3 = SimpleExpSmoothing(y3)
        ets3 = Holt(y3)
        # ets3 = ExponentialSmoothing(y3, trend='add', seasonal='add', seasonal_periods=2)
        r3 = ets3.fit()
        p = r3.predict(start=len(hisResult), end=len(hisResult))[len(hisResult)]
    return p


# 采样分析 data: 数据（2000数据） subsetNum：子集大小（94）
def sampling(data, subsetNum):
    # np.random.shuffle(data)
    subset = []
    for i in range(subsetNum):
        subset.append(data[i])

    count = 0
    for i in range(subsetNum):
        if(subset[i] == 1):
            count = count + 1
    return count/subsetNum


# 预测权重计算 V3
def weight_cal(hisResult, p, s):
    if(len(hisResult) == 0):
        a = 0
    else:
        a_deno1 = 0
        for i in range(len(hisResult)):
            a_deno1 = a_deno1 + (abs(p[i] - hisResult[i]) / hisResult[i])
        a_deno1_avg = a_deno1 / len(hisResult)
        a_deno2 = abs(p[len(p) - 1] - s[len(s) - 1]) / s[len(s) - 1]
        temp = 1 / (a_deno1_avg + a_deno2)
        a = np.exp(-1 / temp)
    return a


# 预测权重计算 V4
# def weight_cal(hisResult, p, s):
#     if(len(hisResult) == 0):
#         a = 0
#     else:
#         a_deno1 = 0
#         for i in range(len(hisResult)):
#             a_deno1 = a_deno1 + (abs(p[i] - hisResult[i]) / hisResult[i])
#         # a_deno1_avg = a_deno1 / len(hisResult)
#         a_deno2 = abs(p[len(p) - 1] - s[len(s) - 1]) / s[len(s) - 1]
#         a_deno_avg = (a_deno1 + a_deno2) / (len(hisResult) + 1)
#         temp = 1 / a_deno_avg
#         a = np.exp(-1 / temp)
#     return a


hisResult = []  # 历史结果 需要初始化吗？
p = []  # 预测值集合
s = []  # 采样值集合
esti_result = []  # 估计值集合
alpha = []
beta = []  # 采样值权重
p_inaccu = []
s_inaccu = []
e_inaccu = []   # 估计值和真实值之间的误差


# att = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
#        350, 600, 1800, 700, 350, 1760, 50, 1320, 5, 350,
#        1200, 1100, 880, 580, 220, 200, 150, 100, 80, 3]   # attacker_behavior 即为每次任务的非法签名数量

# att = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
#        350, 700, 650, 500, 260, 660, 50, 560, 5, 700,
#        600, 550, 510, 430, 380, 270, 150, 100, 80, 10]

# att = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
#        600, 700, 650, 400, 260, 660, 50, 560, 5, 700,
#        600, 550, 500, 430, 380, 270, 150, 100, 80, 10]

# att = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
#        600, 700, 650, 400, 260, 660, 50, 560, 5, 700,
#        600, 550, 500, 450, 400, 350, 300, 250, 200, 150]

# att = [600, 700, 650, 400, 260, 660, 50, 560, 5, 700,
#        50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
#        600, 550, 500, 430, 380, 270, 150, 100, 80, 10]

# att = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
#         600, 550, 500, 430, 380, 270, 150, 100, 80, 10,
#        600, 700, 650, 400, 260, 660, 50, 560, 5, 700]

# att = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 50, 720, 680, 660, 510, 250, 1000, 40, 800, 750, 700, 650, 600, 550]   # attacker_behavior 即为每次任务的非法签名数量

# att = [10, 40, 70, 100, 130, 160, 190, 220, 250, 280, 310, 340, 370, 400, 430, 460, 490, 520, 550, 580, 610, 640, 670, 700,
#        600, 750, 500, 330, 380, 270, 450, 100, 80, 30, 400, 550, 500, 730, 380, 270, 150, 300, 80, 10, 560, 380, 120, 360]

att = [130, 160, 190, 220, 250, 280, 310, 340, 370, 400, 430, 460, 490, 520, 550, 580, 610, 640, 670, 700,
       380, 270, 450, 100, 80, 30, 400, 550, 500, 730, 380, 270, 150, 300, 80, 10, 560, 380, 120, 360]


# att = [2000, 50, 1350, 200, 1700, 150, 1430, 2500, 125, 250, 1000]

for i in range(len(att)):
    batch_data = batch_request_generating(2000, att[i])
    print("第", i, "次任务攻击者投入的非法签名数量为：", att[i])
    print("第", i, "次任务攻击者造成的非法签名占比为：", att[i]/2000)

    # 第一步：预测分析，获取预测值
    # p_t = forecast(hisResult, 0.5)
    p_t = forecast(hisResult)
    p.append(p_t)
    # 第二步：采样分析，获取采样值
    np.random.shuffle(batch_data)
    s_t_1 = sampling(batch_data, 42)
    s.append(s_t_1)
    print("第", i, "次任务的预测值为：", p[i])
    print("第", i, "次任务的采样值为：", s[i])


    #  第三步：重采样，获取更精确的采样值
    print("第", i, "次任务的预测值和采样值的距离为：", (abs(p[i] - s[i]) / s[i]))
    if (abs(p[i]-s[i])/s[i]) > 0.7:  # 这里大于的值为人工设置的一个阈值
        s_t_2 = sampling(batch_data, 92)
        # s_t = sampling(batch_data, 861)
        # print("第", i, "次任务的预测值和采样值的距离为：", (abs(p[i]-s[i])/s[i]))
        s[i] = s_t_2
        print("第", i, "次任务重新采样的采样值为：", s[i], "-----------------------")
        print("第", i, "次任务的预测值和重采样值的距离为：", (abs(p[i] - s[i]) / s[i]))


    #  第四步：预测可信度（即预测权重）计算 & 第五步：采样可信度（即采样权重）计算
    alpha.append(weight_cal(hisResult, p, s))  # 当前预测权重
    beta.append(1 - alpha[i])
    print("第", i, "次任务的预测值的权重α为：", alpha[i])
    print("第", i, "次任务的采样值的权重β为：", beta[i])


    #  第六步：加权求和计算估计值
    e = alpha[i] * p[i] + beta[i] * s[i]
    esti_result.append(e)
    print("第", i, "次任务的估计值为：", esti_result[i])

    # 输出一下预测值、采样值、估计值和真实值之间的误差
    p_ina = (p[i] - (att[i]/2000)) / (att[i]/2000)
    p_inaccu.append(p_ina)

    s_ina = (s[i] - (att[i] / 2000)) / (att[i] / 2000)
    s_inaccu.append(s_ina)

    e_ina = (esti_result[i] - (att[i]/2000)) / (att[i]/2000)
    e_inaccu.append(e_ina)
    print("第", i, "次任务的预测值和真实值之间的误差为：", round(100*(p[i] - (att[i]/2000)) / (att[i]/2000), 2), "%")
    print("第", i, "次任务的第一次采样值和真实值之间的误差为：", round(100*(s_t_1 - (att[i]/2000)) / (att[i]/2000), 2), "%")
    print("第", i, "次任务的第二次采样值和真实值之间的误差为：", round(100*(s[i] - (att[i]/2000)) / (att[i]/2000), 2), "%")
    print("第", i, "次任务的估计值和真实值之间的误差为：", round(100*e_ina, 2), "%")

    # 历史值的记录，本来这个值应该由贡献点2的组合式算法求出，这里为了验证功能，直接复制
    hisResult.append(att[i] / 2000)
    print("【待实现】基于强化学习的组合算法计算出第", i, "次非法签名占比为：", att[i] / 2000)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# 两个阶段，增加+随机波动
p_mse_1 = mean_squared_error(p[0:19], hisResult[0:19])
s_mse_1 = mean_squared_error(s[0:19], hisResult[0:19])
e_mse_1 = mean_squared_error(esti_result[0:19], hisResult[0:19])
print("预测值的MSE为：", p_mse_1)
print("采样值的MSE为：", s_mse_1)
print("估计值的MSE为：", e_mse_1)

data_1 = [p_mse_1, s_mse_1, e_mse_1]
# plt.figure(dpi=300)
# plt.bar(range(len(data_1)), data_1, color='palegreen')
# plt.show()

p_mse_2 = mean_squared_error(p[20:39], hisResult[20:39])
s_mse_2 = mean_squared_error(s[20:39], hisResult[20:39])
e_mse_2 = mean_squared_error(esti_result[20:39], hisResult[20:39])
print("预测值的MSE为：", p_mse_2)
print("采样值的MSE为：", s_mse_2)
print("估计值的MSE为：", e_mse_2)

data_2 = [p_mse_2, s_mse_2, e_mse_2]

p_mse = mean_squared_error(p, hisResult)
s_mse = mean_squared_error(s, hisResult)
e_mse = mean_squared_error(esti_result, hisResult)

labels = ['规律1', '规律2', '综合']
p_values = [p_mse_1, p_mse_2, p_mse]
s_values = [s_mse_1, s_mse_2, s_mse]
e_values = [e_mse_1, e_mse_2, e_mse]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, p_values, width, label='预测值')
rects2 = ax.bar(x, s_values, width, label='采样值')
rects3 = ax.bar(x + width, e_values, width, label='估计值')

ax.set_ylabel('MSE值')
ax.set_title('预测、采样和估计MSE值对比')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()

data = []
data.append(p_values)
data.append(s_values)
data.append(e_values)
header = ['预测值', '采样值', '估计值']
with open('mse_data.csv', 'w', newline='') as file:
    wri = csv.writer(file)
    wri.writerow(header)
    wri.writerows(data)


# 画一组图 非法请求占比
plt.figure(figsize=(10, 8), dpi=300)
ax1 = plt.subplot(2, 2, 1)
plt.plot(hisResult, color='black', marker='o', linestyle='dashed', label='True values')
plt.plot(p, color='palegreen', marker='d', linestyle='dashed', label='Predicted values')
plt.xlabel('实验次数')
plt.ylabel('非法请求占比')
plt.title("对比实验")
plt.legend()

ax2 = plt.subplot(2, 2, 2)
plt.plot(hisResult, color='black', marker='o', linestyle='dashed', label='True values')
plt.plot(s, color='deepskyblue', marker='P', linestyle='dashed', label='Sampling values')
plt.xlabel('实验次数')
plt.ylabel('非法请求占比')
plt.title("对比实验")
plt.legend()

ax3 = plt.subplot(2, 2, 3)
plt.plot(hisResult, color='black', marker='o', linestyle='dashed', label='True values')
plt.plot(esti_result, color='crimson', marker='s', linestyle='dashed', label='Estimated values')
plt.xlabel('实验次数')
plt.ylabel('非法请求占比')
plt.title("对比实验")
plt.legend()

ax4 = plt.subplot(2, 2, 4)
plt.plot(hisResult, color='black', marker='o', linestyle='dashed', label='True values')
plt.plot(p, color='palegreen', marker='d', linestyle='dashed', label='Predicted values')
plt.plot(s, color='deepskyblue', marker='P', linestyle='dashed', label='Sampling values')
plt.plot(esti_result, color='crimson', marker='s', linestyle='dashed', label='Estimated values')
plt.xlabel('实验次数')
plt.ylabel('非法请求占比')
plt.title("对比实验")
plt.legend()

plt.show()


data = []
data.append(hisResult)
data.append(p)
data.append(s)
data.append(esti_result)
data_array = np.array(data)
header = ['真值', '预测值', '采样值', '估计值']
with open('estimate_data.csv', 'w', newline='') as file:
    wri = csv.writer(file)
    wri.writerow(header)
    wri.writerows(data_array.T)



# 画权重变化图
plt.figure(dpi=300)
plt.plot(alpha, color='mediumspringgreen', marker='o', linestyle='dashed', label='Alpha values')
plt.plot(beta, color='mediumslateblue', marker='d', linestyle='dashed', label='Beta values')
plt.xlabel('实验次数')
plt.ylabel('权重值')
plt.title("权重变化情况")
plt.legend()
plt.show()

data = []
data.append(alpha)
data.append(beta)
data_array = np.array(data)
header = ['alpha值', 'beta值']
with open('weights_data.csv', 'w', newline='') as file:
    wri = csv.writer(file)
    wri.writerow(header)
    wri.writerows(data_array.T)

# 画一组误差图
plt.figure(figsize=(10, 10), dpi=300)

p_inaccu_percent = [i*100 for i in p_inaccu]
ax1 = plt.subplot(3, 1, 1)
plt.plot(p_inaccu_percent, color='slateblue', marker='o', linestyle='dashed', label='Predicted Error rates')
# ax1 = brokenaxes(ylims=((0, 100), (1500, 2000)), despine=False, hspace=0.05, d=0.01)
plt.xlabel('实验次数')
plt.ylabel('误差率(%)')
plt.title("预测值误差情况")
plt.legend()

s_inaccu_percent = [i*100 for i in s_inaccu]
ax2 = plt.subplot(3, 1, 2)
plt.plot(s_inaccu_percent, color='darkcyan', marker='o', linestyle='dashed', label='Sampling Error rates')
plt.xlabel('实验次数')
plt.ylabel('误差率(%)')
plt.title("采样值误差情况")
plt.legend()

e_inaccu_percent = [i*100 for i in e_inaccu]
ax3 = plt.subplot(3, 1, 3)
plt.plot(e_inaccu_percent, color='darkorange', marker='o', linestyle='dashed', label='Estimated Error rates')
plt.xlabel('实验次数')
plt.ylabel('误差率(%)')
plt.title("估计值误差情况")
plt.legend()

plt.show()

count_p = 0
count_s = 0
count_e = 0
for item in p_inaccu_percent:
    if item >= -20 and item <= 20:
        count_p = count_p + 1

for item in s_inaccu_percent:
    if item >= -20 and item <= 20:
        count_s = count_s + 1

for item in e_inaccu_percent:
    if item >= -20 and item <= 20:
        count_e = count_e + 1

print("预测值正负误差20%满足条件点数为：", count_p)
print("采样值正负误差20%满足条件点数为：", count_s)
print("估计值正负误差20%满足条件点数为：", count_e)

count_p = 0
count_s = 0
count_e = 0
for item in p_inaccu_percent:
    if item >= -15 and item <= 15:
        count_p = count_p + 1

for item in s_inaccu_percent:
    if item >= -15 and item <= 15:
        count_s = count_s + 1

for item in e_inaccu_percent:
    if item >= -15 and item <= 15:
        count_e = count_e + 1

print("预测值正负误差15%满足条件点数为：", count_p)
print("采样值正负误差15%满足条件点数为：", count_s)
print("估计值正负误差15%满足条件点数为：", count_e)

data = []
data.append(p_inaccu_percent)
data.append(s_inaccu_percent)
data.append(e_inaccu_percent)
data_array = np.array(data)
header = ['Predicted Error rates', 'Sampling Error rates', 'Estimated Error rates']
with open('error_data.csv', 'w', newline='') as file:
    wri = csv.writer(file)
    wri.writerow(header)
    wri.writerows(data_array.T)