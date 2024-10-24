import base64
import io
import requests
import torch
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import time

from models.Nets import CNN
from models.Fed import getDis, getAlpha, getTauI, normalization
from utils.options import args_server_parser
from utils.models import hexToStateDict, stateDictToHex

# 解析服务器参数
args = args_server_parser()

# 初始化全局模型
def initNetGlob():
    if args.model == 'cnn':
        net_glob = CNN(num_classes=args.num_classes, num_channels=args.num_channels)
    else:
        exit('Error: unrecognized model')
    return net_glob

# 初始化 Flask 应用和 SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# 初始化全局模型
net_glob = initNetGlob()

# 全局状态，用于管理多个节点的状态
globalState = {
    'n_d': 0,  # 节点数
    's': [
        [], [], [], []
    ],  # 存储分数和距离等信息
    't': [],  # 时间戳
    'scores': [],  # 各个节点的得分
    'uid': {}  # 节点的 UID 映射
}

# 模型聚合的结果
weights = []

# 合并本地模型到全局模型
def merge(uid, address, data):
    print("Merging local model from node:", address)
    
    # 定义真实标签分布和预测分布
    p = [1, 0, 0]  # 真实标签分布 (one-hot 编码)
    q = [0.8, 0.1, 0.1]  # 模型预测的概率分布
    
    # 计算 alpha
    alpha = getAlpha(1, int(time.time()), data['t0'], 0.003, 1, 
                    data['uid'], data['n_d'], data['s'], p, q)
    
    if alpha == 0:
        return
    
    localStateDict = data['local_state_dict']
    globStateDict = data['global_state_dict']
    
    # 更新全局模型参数
    for k in globStateDict.keys():
        globStateDict[k] = (1 - alpha) * globStateDict[k] + alpha * localStateDict[k]

    stateDictHex = stateDictToHex(globStateDict)
    s = normalization(data['s'])
    score = getTauI(uid, data['n_d'], s, p, q)
    
    return {
        'model_state_hex': stateDictHex,
        'score': score,
        'address': address,
        'cur_global_state_dict': globStateDict
    }

# 接收并合并节点的本地模型
@app.route('/newLocalModel/<address>', methods=['POST'])
def newLocalModel(address):
    data = request.json
    localStateDict = hexToStateDict(data['local_model_hex'])
    
    global net_glob
    globalStateDict = net_glob.state_dict()  # 获取当前全局模型的状态字典
    
    uid = globalState['uid'][address]  # 获取节点的 UID
    globalState['s'][3][uid] = getDis(globalStateDict, localStateDict)  # 更新距离

    # 合并本地模型到全局模型
    res = merge(uid, address, {
        "local_state_dict": localStateDict,
        "global_state_dict": globalStateDict,
        "s": globalState['s'],
        "n_d": globalState['n_d'],
        "uid": uid,
        "t0": globalState['t'][uid]
    })

    if res:
        # 更新全局状态
        globalState['t'][uid] = time.time()
        globalState['s'][1][uid] = float(res['score'])
        net_glob.load_state_dict(hexToStateDict(res['model_state_hex']))  # 更新全局模型
        print(f"Model from {address} successfully merged into global model.")
    
    return '1'

# 返回或上传全局模型
@app.route('/newGlobalModel', methods=['GET', 'POST'])
def newGlobalModel():
    global net_glob
    if request.method == 'POST':
        # 从客户端接收到的全局模型
        data = request.json
        globalModelStateDict = hexToStateDict(data['global_model_hex'])
        net_glob.load_state_dict(globalModelStateDict)
        return '1'
    elif request.method == 'GET':
        # 返回当前全局模型的十六进制状态
        if net_glob is None:
            print("No global model available.")
            return jsonify({"error": "No global model available"}), 500
        
        state_dict_hex = stateDictToHex(net_glob.state_dict())
        print(f"Returning global model state: {state_dict_hex[:50]}...")  # 打印部分十六进制以确认返回值
        return jsonify({'global_model_hex': state_dict_hex})

# 注册新的节点
def register(address, dataSize):
    global globalState
    if address in globalState['uid']:
        return False  # 节点已存在

    uid = globalState['n_d']  # 分配一个新的 UID
    globalState['uid'][address] = uid
    globalState['n_d'] += 1  # 更新节点计数
    globalState['scores'].append(0)
    globalState['t'].append(time.time())  # 保存当前时间戳
    globalState['s'][0].append(dataSize)  # 记录数据大小
    globalState['s'][1].append(0.5)  # 默认得分
    globalState['s'][2].append(1)  # 默认 Tau 值
    globalState['s'][3].append(0)  # 默认距离
    return True

# 处理节点的注册请求
@app.route('/register', methods=['POST'])
def handleRegister():
    data = request.json
    address = data['address']
    dataSize = int(data['data_size'])
    if register(address, dataSize):
        print(f"Node {address} registered successfully with data size {dataSize}.")
        return '1'
    else:
        print(f"Node {address} already registered.")
        return '0'

# 返回全局模型的元信息 (如模型类型和输入输出结构)
@app.route("/getTrainInfo", methods=['GET'])
def getModelType():
    return {
        'model_name': args.model,
        'num_classes': args.num_classes,
        'num_channels': args.num_channels
    }

# 启动服务器
if __name__ == '__main__':
    print(f"Starting server on port {args.port}...")
    socketio.run(app, port=args.port, allow_unsafe_werkzeug=True)
