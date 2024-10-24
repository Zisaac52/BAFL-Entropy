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
def calculate_permutation_entropy(time_series, m=3, delay=1):
    """
    计算排列熵
    time_series: 时间序列数据
    m: 嵌入维度
    delay: 时间延迟
    """
    n = len(time_series)
    if n < m:
        return 0
    
    # 生成所有可能的排列模式
    patterns = {}
    for i in range(n - (m-1)*delay):
        # 获取长度为m的子序列
        pattern = []
        for j in range(m):
            pattern.append(time_series[i + j*delay])
        
        # 获取排列顺序
        sorted_idx = sorted(range(len(pattern)), key=lambda k: pattern[k])
        pattern_str = ''.join(map(str, sorted_idx))
        
        # 统计模式出现次数
        patterns[pattern_str] = patterns.get(pattern_str, 0) + 1
    
    # 计算每个模式的概率并计算熵
    total_patterns = sum(patterns.values())
    entropy = 0
    for count in patterns.values():
        p = count / total_patterns
        entropy -= p * math.log(p)
    
    # 归一化
    max_entropy = math.log(math.factorial(m))
    return entropy / max_entropy if max_entropy != 0 else 0

# 根据熵权法取得当前指标的权重
def getWk(N_D, s, s_i):
    """
    使用排列熵计算权重
    W_j = (1 - H_perm(X_j)) / Σ(1 - H_perm(X_i))
    """
    # 计算每个指标的排列熵
    entropy = calculate_permutation_entropy(s_i, m=3)
    
    # 计算所有指标的排列熵之和
    sum_entropy = 0
    for i in range(len(s)):
        sum_entropy += calculate_permutation_entropy(s[i], m=3)
    
    # 计算权重
    if sum_entropy == 0:
        return 1.0 / len(s)
    return (1 - entropy) / (len(s) - sum_entropy)


def getTauI(i, N_D, s):
    """
    计算综合得分
    考虑时间序列的复杂度
    """
    sum_score = 0
    for k in range(len(s)):
        weight = getWk(N_D, s, s[k])
        sum_score += weight * s[k][i]
    return sum_score

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


def getAlpha(kexi, t, t0, theta, R0, i, N_D, s):
    """
    计算聚合权重
    结合排列熵和时间衰减
    """
    # 归一化数据
    s = normalization(s)
    
    # 计算时间衰减和基于排列熵的权重
    time_weight = getR(t, t0, theta, R0)
    entropy_weight = getTauI(i, N_D, s)
    
    return kexi * time_weight * entropy_weight