#!/usr/bin/env python
# coding=utf-8
'''
@Author: JK211
@Email: jerryren2884@gmail.com
@Date: 2023-11-27 11:59
@LastEditor: JK211
LastEditTime: 2024-1-20
@Discription:  我们利用一个01序列来模拟批量请求，其中1代表非法请求，0代表合法请求
               We model batch requests using a sequence of 01s, where 1 represent illegal requests and 0 represent legal requests
@Environment: python 3.7
'''
import numpy as np
import random as rand


def batch_request_generating(batch_size, invalid_size):
    Sigs2beVefi = np.zeros(batch_size, int)
    for i in range(invalid_size):
        positive_position = rand.randrange(batch_size)

        while Sigs2beVefi[positive_position] == 1:
            positive_position = rand.randrange(batch_size)

        Sigs2beVefi[positive_position] = 1

    return Sigs2beVefi


batch = batch_request_generating(2000, 20).tolist()
print(batch)
print(type(batch))
