#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy

import torch
import math

# 计算交叉熵
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
# 计算交叉熵
def calculate_cross_entropy(p, q):
    cross_entropy = 0
    for i in range(len(p)):
        if q[i] > 0:
            cross_entropy += p[i] * math.log(q[i])
    return -cross_entropy

# 根据交叉熵计算权重
def calculate_weights_cross_entropy(entropies):
    sum_entropy = sum([1 - H for H in entropies])
    if sum_entropy == 0:
        return [1.0 / len(entropies)] * len(entropies)
    weights = [(1 - H) / sum_entropy for H in entropies]
    return weights


# def getTauI(i, N_D, s):
#     sum = 0
#     for k in range(len(s)):
#         sum += getWk(N_D, s, s[k]) * s[k][i]
#     return sum
# 修改 getTauI 函数，使用交叉熵计算权重
def getTauI(i, N_D, s, p, q):
    """
    计算客户端的综合评分
    i: 当前客户端索引
    N_D: 客户端总数
    s: 评分列表
    p: 真实标签分布
    q: 预测概率分布
    """
    # 检查输入数据的有效性
    if not s or len(s) == 0:
        return 0.0
    
    # 确保所有评分列表长度一致
    list_length = len(s[0])
    if i >= list_length:
        print(f"Warning: Index {i} out of range for score list length {list_length}")
        return 0.0
    
    # 为每个维度计算交叉熵
    cross_entropies = []
    for k in range(len(s)):
        if len(s[k]) > 0:  # 确保该维度有数据
            # 使用归一化后的分数作为预测分布
            normalized_scores = normalization([s[k]])[0]
            if i < len(normalized_scores):
                cross_entropy = calculate_cross_entropy(p, q)
                cross_entropies.append(cross_entropy)
    
    if not cross_entropies:  # 如果没有有效的交叉熵值
        return 0.0
    
    # 计算权重
    weights = []
    total_weight = 0
    for entropy in cross_entropies:
        weight = 1 / (entropy + 1e-10)  # 添加小值避免除零
        weights.append(weight)
        total_weight += weight
    
    # 归一化权重
    if total_weight == 0:
        return 0.0
        
    normalized_weights = [w/total_weight for w in weights]
    
    # 计算加权得分
    weighted_score = 0
    for k in range(min(len(normalized_weights), len(s))):
        if k < len(s) and i < len(s[k]):
            weighted_score += normalized_weights[k] * s[k][i]
    
    return weighted_score

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


def getAlpha(kexi, t, t0, theta, R0, i, N_D, s, p, q):
    # 归一化 s
    s = normalization(s)
    
    # 计算 R(t)
    r_value = getR(t, t0, theta, R0)
    
    # 计算 TauI
    tau_i = getTauI(i, N_D, s, p, q)
    
    # 返回 alpha
    return kexi * r_value * tau_i
