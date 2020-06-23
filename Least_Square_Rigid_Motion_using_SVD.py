# -*- coding: utf-8 -*-

"""
运行环境: Python36_Computing
内容： 实验 Least Square Rigid Motion using SVD
准备简单实验数据
1. 生成一个简单的正方形“点云” （0，0）， （1，0), (1,1), (0,1) ,用一个4x3 的矩阵表示
2. 利用算转矩阵和平移矩阵，将P逆时针旋转90度，向x正方向平移2

"""

import numpy as np
import scipy

# 计算两片点云的centroid, 假设所有点的权重都是1


# 给定旋转角度
theta = 90
theta = (theta / 180.0) * np.pi
print("Radian: %.3f" % theta)

# 点云P
P = np.array([ [0, 0, 0],
               [1, 0, 0],
               [1, 1, 0],
               [0, 1, 0]])
P = P.T
print("点云,", P.shape)
print(P)

# 真实旋转矩阵（绕z轴逆时针旋转90度
real_rotation = np.array([ [np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0],
                           [0, 0,1]
                           ])
# 点云Q
Q = np.zeros(P.shape)
for i in range(P.shape[1]):
    Q[i,:] = np.dot(real_rotation, P[i,:])

Q = np.around(Q)
print("点云Q,", Q.shape)
print(Q)

# 平移（2， 0， 0）
real_t = np.array([ 2, 0, 0])
Q = Q + real_t
print("最终Q:")
print(Q)


# 数据准备完毕，现在假设已知P， Q， 计算R和t，看是否和real_rotation, real_t 一致
def compute_centroid(X):
    centroid = np.zeros(3)
    for i in range(X.shape[0]):
        centroid += X[i, :]
    centroid = centroid / X.shape[0]
    return centroid


p_centroid = compute_centroid(P)
q_centroid = compute_centroid(Q)


print("P centroid: ", p_centroid)
print("Q centroid: ", q_centroid)

X = P - p_centroid
Y = Q - q_centroid

W = np.identity(4)

S = np.dot(np.dot(X, W), Y.T)
print(S)