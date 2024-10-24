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
def calculate_KL_divergence(p, q):
    """
    计算 KL 散度: D_KL(p||q) = Σ p(x)log(p(x)/q(x))
    p: 真实分布
    q: 预测分布
    """
    try:
        # 确保 p 和 q 是有效的概率分布
        if not (isinstance(p, list) and isinstance(q, list)):
            print(f"Invalid input types: p={type(p)}, q={type(q)}")
            return 1.0
        
        if len(p) != len(q):
            print(f"Mismatched dimensions: p={len(p)}, q={len(q)}")
            return 1.0
        
        # 添加平滑因子，避免除零
        epsilon = 1e-10
        kl_div = 0
        
        for i in range(len(p)):
            p_i = p[i] + epsilon
            q_i = q[i] + epsilon
            if p_i > 0:
                kl_div += p_i * math.log(p_i / q_i)
        
        return max(0, kl_div)  # 确保返回非负值
        
    except Exception as e:
        print(f"Error in calculate_KL_divergence: {e}")
        print(f"p={p}, q={q}")
        return 1.0  # 返回较大的 KL 散度值作为惩罚

# 计算权重
def calculate_weights_KL(kl_divergences):
    """
    根据 KL 散度计算权重
    W_j = (1 - D_KL(p||q_j)) / Σ(1 - D_KL(p||q_i))
    """
    weights = []
    total = 0
    for div in kl_divergences:
        weight = 1 - div  # KL 散度越小，权重越大
        weights.append(weight)
        total += weight
    
    # 归一化权重
    if total > 0:
        weights = [w/total for w in weights]
    return weights

def getTauI(i, N_D, s, p=None, q=None):
    """
    使用 KL 散度计算客户端权重
    i: 当前客户端索引
    N_D: 客户端总数
    s: 评分列表
    p: 真实分布 (可选)
    q: 预测分布 (可选)
    """
    try:
        # 如果没有提供 p 和 q，使用默认值
        if p is None:
            p = [1, 0, 0]  # 默认 one-hot 编码
        if q is None:
            q = [0.8, 0.1, 0.1]  # 默认预测分布
        
        # 计算 KL 散度
        kl_div = calculate_KL_divergence(p, q)
        
        # 计算权重：KL 散度越小，权重越大
        weight = 1 / (1 + kl_div)  # 使用 1/(1+KL) 确保权重为正且在 0-1 之间
        
        # 计算加权得分
        weighted_score = 0
        for k in range(len(s)):
            weighted_score += weight * s[k][i]
        
        return weighted_score
    except Exception as e:
        print(f"Error in getTauI: {e}")
        print(f"Parameters: i={i}, N_D={N_D}, p={p}, q={q}")
        return 0.5  # 返回默认值


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
    计算聚合权重 alpha
    kexi: 系数
    t: 当前时间
    t0: 初始时间
    theta: 衰减系数
    R0: 初始权重
    i: 客户端索引
    N_D: 客户端总数
    s: 评分列表
    """
    # 归一化解决时间戳数值过大导致的权重过小的问题
    s = normalization(s)
    
    # 获取真实分布和预测分布
    p = [1, 0, 0]  # 示例：one-hot 编码
    q = [0.8, 0.1, 0.1]  # 示例：预测分布
    
    # 使用默认的 p 和 q
    return kexi * getR(t, t0, theta, R0) * getTauI(i, N_D, s)
