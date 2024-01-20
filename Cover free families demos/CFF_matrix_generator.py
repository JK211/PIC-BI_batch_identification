#!/usr/bin/env python
# coding=utf-8
'''
@Author: JK211
@Email: jerryren2884@gmail.com
@Date: 2023-11-27 11:59
@LastEditor: JK211
LastEditTime: 2023-11-27
@Discription:  这里实现了一个8-CFF(289,4913)非覆盖族的实例，并将其关联矩阵保存下来
               Here an instance of the 8-CFF(289,4913) non-covering family is realized and its association matrix is preserved
@Environment: python 3.7
'''
import numpy as np
import math

row = 289
col = 4913

M = np.zeros((row, col), dtype=np.int8)

for i in range(int(row**0.5)):
    for j in range(int(row**0.5)):
        r = i*int(row**0.5) + j
        l = 0
        for a in range(int(row**0.5)):
            for b in range(int(row**0.5)):
                for c in range(int(row**0.5)):
                    # print(i, j, l)
                    if (a*i*i + b*i + c) % int(row**0.5) == j:
                        M[r][l] = 1
                    else:
                        M[r][l] = 0
                    l = l + 1

np.save("8-cff289-4913.npy", M)

b = np.load("8-cff289-4913.npy")
np.set_printoptions(threshold = np.inf)
print(b)
