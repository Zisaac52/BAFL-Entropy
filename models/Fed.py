import copy
import torch
import math

# 联邦平均算法，用于聚合客户端模型权重
def FedAvg(w, client_participations, max_participation):
    w_avg = copy.deepcopy(w[0])
    num_clients = len(w)

    for k in w_avg.keys():
        w_avg[k] = torch.zeros_like(w_avg[k])  # 初始化为0

        total_weight = 0
        # 遍历所有客户端，动态计算每个客户端的权重并进行加权平均
        for i in range(num_clients):
            # 动态调整 α 值
            client_alpha = adjust_alpha(client_participations[i], max_participation)
            # 计算权重
            client_weight = getWk(len(w[i][k]), w[i][k], w[i][k], client_participations[i], max_participation)
            # 累加加权的权重
            w_avg[k] += w[i][k] * client_weight
            total_weight += client_weight

        # 计算平均权重
        if total_weight > 0:
            w_avg[k] /= total_weight
    
    return w_avg

# 计算客户端与全局模型之间的距离
def getDis(globalW, w):
    sumDis = 0
    w_avg = copy.deepcopy(w)
    for i in w_avg.keys():
        sumDis += torch.norm(w[i] - globalW[i], 2)
    return pow(float(sumDis), 0.5)

# 计算概率
def getP(s_k, s_k_i):
    if s_k_i == 0:
        return 0
    sum_s_k = sum(s_k)
    return s_k_i / sum_s_k

# 动态调整 α 参数
def adjust_alpha(client_participation, max_participation, base_alpha=1, max_alpha=3, min_alpha=0.5):
    """
    动态调整 α 参数。
    
    参数：
    - client_participation: 当前客户端参与的次数
    - max_participation: 最大参与次数，用于归一化
    - base_alpha: 基础 α 值
    - max_alpha: α 的最大值
    - min_alpha: α 的最小值
    
    返回：
    - 调整后的 α 值
    """
    # 将参与次数标准化到 [0, 1] 区间
    participation_ratio = client_participation / max_participation

    # 根据参与度线性调整 α 值
    if participation_ratio >= 0.5:
        alpha = base_alpha + (max_alpha - base_alpha) * (participation_ratio - 0.5) * 2
    else:
        alpha = base_alpha - (base_alpha - min_alpha) * (0.5 - participation_ratio) * 2
    
    # 确保 α 在 min_alpha 和 max_alpha 之间
    return max(min(alpha, max_alpha), min_alpha)

# 计算 Rényi 熵
def getRenyiEntropy(N_D, s_k, alpha=2):
    if alpha == 1:
        # α = 1 时，Rényi 熵退化为 Shannon 熵，特殊处理
        sum_entropy = 0
        for i in range(N_D):
            p = getP(s_k, s_k[i])
            if p == 0:
                continue
            sum_entropy += p * math.log(p)
        return -1.0 * (1 / math.log(N_D)) * sum_entropy
    else:
        # Rényi 熵公式计算
        sum_p_alpha = 0
        for i in range(N_D):
            p = getP(s_k, s_k[i])
            if p == 0:
                continue
            sum_p_alpha += pow(p, alpha)  # 计算 p^alpha
        renyi_entropy = (1 / (1 - alpha)) * math.log(sum_p_alpha)  # 公式中的 log 部分
        return renyi_entropy

# 计算权重
def getWk(N_D, s, s_i, client_participation, max_participation):
    # 动态调整 α
    alpha = adjust_alpha(client_participation, max_participation, base_alpha=1, max_alpha=3, min_alpha=0.5)
    
    sum_entropy = 0
    for i in range(len(s)):
        sum_entropy += getRenyiEntropy(N_D, s[i], alpha)  # 计算所有指标的熵值和
    return (1 - getRenyiEntropy(N_D, s_i, alpha)) / (len(s) - sum_entropy)  # 返回当前指标的权重

# 指标加权平均
def getTauI(i, N_D, s, client_participation, max_participation):
    sum = 0
    for k in range(len(s)):
        sum += getWk(N_D, s, s[k], client_participation, max_participation) * s[k][i]
    return sum

# 根据牛顿冷却法取得当前模型的权重
def getR(t, t0, theta, R0):
    return R0 * pow(math.e, -1 * (theta * (t - t0)))

# 数据归一化
def normalization(s):
    res = []
    for k in range(len(s)):
        res.append([])
        sum = sum(s[k])
        if sum == 0:
            return 0
        for i in range(len(s[k])):
            res[k].append(s[k][i] / sum)
    return res

# 最终权重计算
def getAlpha(kexi, t, t0, theta, R0, i, N_D, s, client_participation, max_participation):
    # 归一化解决时间戳数值过大导致的熵权过小的问题
    s = normalization(s)
    return kexi * getR(t, t0, theta, R0) * getTauI(i, N_D, s, client_participation, max_participation)
