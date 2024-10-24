#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy

import torch
import math


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
# 使用交叉熵进行加权的联邦平均算法
# def FedAvgCrossEntropy(w, entropies):
#     """
#     w: 客户端的模型权重列表
#     entropies: 客户端的交叉熵列表
#     """
#     weights = calculate_weights_cross_entropy(entropies)
#     w_avg = copy.deepcopy(w[0])

#     for k in w_avg.keys():
#         w_avg[k] = torch.zeros_like(w_avg[k])  # 初始化为0
#         for i in range(len(w)):
#             w_avg[k] += w[i][k] * weights[i]
    
    return w_avg

def getDis(globalW, w):
    sumDis = 0
    w_avg = copy.deepcopy(w)
    for i in w_avg.keys():
        sumDis += torch.norm(w[i] - globalW[i], 2)

    return pow(float(sumDis), 0.5)


# def getP(s_k, s_k_i):
#     if s_k_i == 0:
#         return 0
#     sum_s_k = 0
#     for cur_s_k_i in s_k:
#         sum_s_k += cur_s_k_i
#     return s_k_i / sum_s_k


# def getEk(N_D, s_k):
#     sum = 0
#     for i in range(N_D):
#         p = getP(s_k, s_k[i])
#         if p == 0:
#             continue
#         sum += p * math.log(p)
#     return -1.0 * (1 / math.log(N_D)) * sum


# # 根据熵权法取得当前指标的权重
# def getWk(N_D, s, s_i):
#     sum = 0
#     for i in range(len(s)):
#         sum += getEk(N_D, s[i])
#     return (1 - getEk(N_D, s_i)) / (len(s) - sum)

# # 计算 Rényi 熵
# def getRenyiEntropy(N_D, s_k, alpha=2):
#     if alpha == 1:
#         # α = 1 时，Rényi 熵退化为 Shannon 熵，特殊处理
#         sum_entropy = 0
#         for i in range(N_D):
#             p = getP(s_k, s_k[i])
#             if p == 0:
#                 continue
#             sum_entropy += p * math.log(p)
#         return -1.0 * (1 / math.log(N_D)) * sum_entropy
#     else:
#         # Rényi 熵公式计算
#         sum_p_alpha = 0
#         for i in range(N_D):
#             p = getP(s_k, s_k[i])
#             if p == 0:
#                 continue
#             sum_p_alpha += pow(p, alpha)  # 计算 p^alpha
#         renyi_entropy = (1 / (1 - alpha)) * math.log(sum_p_alpha)  # 公式中的 log 部分
#         return renyi_entropy

# # 计算权重
# def getWk(N_D, s, s_i, alpha=1):
#     sum_entropy = 0
#     for i in range(len(s)):
#         sum_entropy += getRenyiEntropy(N_D, s[i], alpha)  # 计算所有指标的熵值和
#     return (1 - getRenyiEntropy(N_D, s_i, alpha)) / (len(s) - sum_entropy)  # 返回当前指标的权重
# 计算交叉熵 H(p, q)
def calculate_cross_entropy(p, q):
    """
    计算交叉熵 H(p, q)
    p: 真实分布 (ground truth, 通常是 one-hot 编码)
    q: 预测分布 (模型输出的概率)
    """
    cross_entropy = 0
    for i in range(len(p)):
        if q[i] > 0:  # 避免 log(0) 错误
            cross_entropy += p[i] * math.log(q[i])
    
    return -cross_entropy


# 计算权重 W_j
def calculate_weights_cross_entropy(entropies):
    """
    计算基于交叉熵的权重 W_j
    entropies: 每个客户端的交叉熵值
    """
    sum_entropy = sum([1 - H for H in entropies])
    if sum_entropy == 0:
        return [1.0 / len(entropies)] * len(entropies)
    
    weights = [(1 - H) / sum_entropy for H in entropies]
    return weights
# # 指标加权平均
# def getTauI(i, N_D, s):
#     sum = 0
#     for k in range(len(s)):
#         sum += getWk(N_D, s, s[k]) * s[k][i]
#     return sum
# 添加交叉熵版本的 getTauI
def getTauI(i, N_D, s, p, q):
    """
    计算交叉熵版本的 Tau 值。
    i: 当前客户端的索引
    N_D: 客户端数量
    s: 客户端的指标值
    p: 真实分布（每个客户端的数据标签）
    q: 预测分布（每个客户端模型输出的概率）
    """
    cross_entropies = []
    for client_idx in range(N_D):
        cross_entropies.append(calculate_cross_entropy(p[client_idx], q[client_idx]))  # 计算交叉熵
    
    weights = calculate_weights_cross_entropy(cross_entropies)  # 计算权重
    sum_weighted_entropy = 0
    for k in range(N_D):
        sum_weighted_entropy += weights[k] * s[k][i]  # 加权求和

    return sum_weighted_entropy


# 根据牛顿冷却法取得当前模型的权重
def getR(t, t0, theta, R0):
    return R0 * pow(math.e, -1 * (theta * (t - t0)))


def normalization(s):
    res = []
    for k in range(len(s)):
        res.append([])
        # min = s[k][0]
        # max = s[k][0]
        # for i in range(len(s[k])):
        #     if s[k][i] < min:
        #         min = s[k][i]
        #     if s[k][i] > max:
        #         max = s[k][i]
        sum = 0
        for i in range(len(s[k])):
            sum += s[k][i]
        if sum == 0:
            return 0
        # for i in range(len(s[k])):
        #     if max == min:
        #         res[k].append(1/len(s[k]))
        #         continue
        #     res[k].append((s[k][i] - min) / (max - min))
        for i in range(len(s[k])):
            res[k].append(s[k][i] / sum)
    return res


# def getAlpha(kexi, t, t0, theta, R0, i, N_D, s):
#     # 归一化解决时间戳数值过大导致的熵权过小的问题。
#     s = normalization(s)
#     return kexi * getR(t, t0, theta, R0) * getTauI(i, N_D, s)
def getAlpha(kexi, t, t0, theta, R0, i, N_D, s, p, q):
    """
    使用交叉熵计算的动态 Alpha 权重
    kexi: 系数
    t, t0: 时间戳
    theta: 温度参数
    R0: 初始权重
    i: 当前客户端索引
    N_D: 客户端数量
    s: 指标值
    p: 真实分布
    q: 预测分布
    """
    # 归一化
    s = normalization(s)
    
    # 使用交叉熵计算 Tau
    tau_i = getTauI(i, N_D, s, p, q)
    
    return kexi * getR(t, t0, theta, R0) * tau_i