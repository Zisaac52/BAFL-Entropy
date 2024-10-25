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


def getDis(globalW, w):
    sumDis = 0
    w_avg = copy.deepcopy(w)
    for i in w_avg.keys():
        sumDis += torch.norm(w[i] - globalW[i], 2)

    return pow(float(sumDis), 0.5)


def getP(s_k, s_k_i):
    if s_k_i == 0:
        return 0
    sum_s_k = 0
    for cur_s_k_i in s_k:
        sum_s_k += cur_s_k_i
    return s_k_i / sum_s_k


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

# 计算 Tsallis 熵
def getTsallisEntropy(N_D, s_k, q=2):
    """改进的 Tsallis 熵计算"""
    # 计算概率分布
    probabilities = []
    sum_p = 0
    for i in range(N_D):
        p = getP(s_k, s_k[i])
        if p > 0:  # 只考虑正概率
            probabilities.append(p)
            sum_p += p
    
    # 归一化概率
    if sum_p == 0:
        return 0
    probabilities = [p/sum_p for p in probabilities]
    
    if q == 1:
        # Shannon 熵情况
        entropy = 0
        for p in probabilities:
            entropy -= p * math.log(p)
        return entropy / math.log(max(2, N_D))  # 避免 N_D=1 的情况
    else:
        # Tsallis 熵计算
        sum_p_q = sum(pow(p, q) for p in probabilities)
        return (1 - sum_p_q) / (q - 1)

# # 计算权重
# def getWk(N_D, s, s_i, alpha=1):
#     sum_entropy = 0
#     for i in range(len(s)):
#         sum_entropy += getRenyiEntropy(N_D, s[i], alpha)  # 计算所有指标的熵值和
#     return (1 - getRenyiEntropy(N_D, s_i, alpha)) / (len(s) - sum_entropy)  # 返回当前指标的权重
# 计算权重 W_j
def getWkTsallis(N_D, s, s_i, q=2):
    """改进的权重计算"""
    try:
        entropies = []
        for i in range(len(s)):
            entropy = getTsallisEntropy(N_D, s[i], q)
            entropies.append(entropy)
        
        sum_entropy = sum(entropies)
        if sum_entropy == len(s):  # 所有熵都为1的情况
            return 1.0 / len(s)
            
        current_entropy = getTsallisEntropy(N_D, s_i, q)
        weight = (1 - current_entropy) / (len(s) - sum_entropy)
        
        # 确保权重在合理范围内
        return max(0, min(1, weight))
    except:
        return 1.0 / len(s)  # 出错时返回均匀权重
# # 指标加权平均
# def getTauI(i, N_D, s):
#     sum = 0
#     for k in range(len(s)):
#         sum += getWk(N_D, s, s[k]) * s[k][i]
#     return sum
# 修改后的 getTauI 函数
def getTauI(i, N_D, s, q=2):
    """
    使用 Tsallis 熵计算加权平均 Tau。
    
    参数:
    - i: 当前节点索引
    - N_D: 总数据数目
    - s: 节点的指标集合
    - q: Tsallis 熵的参数
    
    返回:
    - 加权平均结果 Tau_i
    """
    sum = 0
    for k in range(len(s)):
        sum += getWkTsallis(N_D, s, s[k], q) * s[k][i]
    return sum

# 根据牛顿冷却法取得当前模型的权重
def getR(t, t0, theta, R0):
    return R0 * pow(math.e, -1 * (theta * (t - t0)))


def normalization(s):
    """改进的归一化处理"""
    if not s or not s[0]:
        return s
    
    res = []
    for k in range(len(s)):
        if not s[k]:
            res.append([])
            continue
            
        sum_values = sum(s[k])
        if sum_values == 0:
            res.append([1.0/len(s[k])] * len(s[k]))  # 均匀分布
        else:
            res.append([v/sum_values for v in s[k]])
    return res


def getAlpha(kexi, t, t0, theta, R0, i, N_D, s):
    # 归一化解决时间戳数值过大导致的熵权过小的问题。
    s = normalization(s)
    return kexi * getR(t, t0, theta, R0) * getTauI(i, N_D, s)
