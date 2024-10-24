#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy

import torch
import math
import numpy as np 

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
def calculate_sample_entropy(time_series, m=2, r=0.2):
    """
    计算样本熵
    time_series: 时间序列数据
    m: 模式长度
    r: 相似度阈值（通常为标准差的0.2倍）
    """
    # 转换输入为numpy数组
    time_series = np.array(time_series, dtype=np.float64)
    
    # 处理特殊情况
    if len(time_series) < m + 1:
        return 0
    
    def _get_matches(data, template, r):
        """计算匹配模式的数量"""
        matches = 0
        for i in range(len(data) - len(template) + 1):
            if np.all(np.abs(data[i:i+len(template)] - template) <= r):
                matches += 1
        return matches

    # 计算标准差
    std = np.std(time_series, ddof=1)  # ddof=1 使用样本标准差
    if std == 0:
        return 0
    
    r = r * std  # 计算阈值
    
    # 计算匹配数
    N = len(time_series)
    B = 0.0  # 匹配数 (m)
    A = 0.0  # 匹配数 (m+1)
    
    # 对每个可能的起始点计算匹配
    for i in range(N - m):
        template_m = time_series[i:i+m]
        template_m1 = time_series[i:i+m+1]
        
        # 计算匹配数
        B += _get_matches(time_series[i+1:], template_m, r)
        A += _get_matches(time_series[i+1:], template_m1, r)
    
    # 避免除零错误
    if B == 0 or A == 0:
        return 0
    
    # 计算样本熵
    return -np.log(A / B)


# 根据熵权法取得当前指标的权重
def getWk(N_D, s, s_i):
    """
    使用样本熵计算权重
    W_j = (1 - H_sample(X_j)) / Σ(1 - H_sample(X_i))
    """
    try:
        # 计算当前指标的样本熵
        entropy = calculate_sample_entropy(s_i, m=2)
        
        # 计算所有指标的样本熵之和
        sum_entropy = 0
        for i in range(len(s)):
            sum_entropy += calculate_sample_entropy(s[i], m=2)
        
        # 避免除零错误
        if sum_entropy == 0:
            return 1.0 / len(s)
        
        # 计算权重：熵值越小，权重越大
        return (1 - entropy) / (len(s) - sum_entropy)
    except Exception as e:
        print(f"Error in getWk: {e}")
        return 1.0 / len(s)  # 返回平均权重作为后备方案


def getTauI(i, N_D, s):
    """
    计算综合得分
    使用基于样本熵的权重
    """
    sum_score = 0
    weights = []
    
    # 计算所有指标的权重
    for k in range(len(s)):
        weight = getWk(N_D, s, s[k])
        weights.append(weight)
        sum_score += weight * s[k][i]
    
    # 打印权重分布，用于调试
    print(f"Indicator weights: {weights}")
    
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
    结合样本熵和时间衰减
    """
    # 归一化数据
    s = normalization(s)
    
    # 计算基于样本熵的权重
    entropy_weight = getTauI(i, N_D, s)
    
    # 计算时间衰减
    time_weight = getR(t, t0, theta, R0)
    
    # 打印调试信息
    print(f"Entropy weight: {entropy_weight}, Time weight: {time_weight}")
    
    return kexi * time_weight * entropy_weight
