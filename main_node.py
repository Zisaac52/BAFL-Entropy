import time
import requests
import torch
from node import Node
from torchvision import datasets, transforms
from utils.dataset import test_img
from utils.options import args_node_parser
from utils.sampling import iid, non_iid
from models.Nets import CNN
from utils.models import stateDictToHex, hexToStateDict

# 解析命令行参数
args = args_node_parser()

# 注册节点到服务器
def register(dataSize):
    resp = requests.post(f'http://localhost:{args.port}/register', json={
        "address": args.address,
        "data_size": dataSize
    })
    # 打印调试信息
    print(f"Register Response: Status Code {resp.status_code}, Response Text: {resp.text}")
    return resp.text == '1'

# 获取最新的全局模型
def getNetGlob():
    # 获取模型结构和参数信息
    resp = requests.get(f'http://localhost:{args.port}/getTrainInfo')
    trainInfo = resp.json()
    
    # 根据模型类型初始化
    if trainInfo['model_name'] == "cnn":
        model = CNN(num_channels=trainInfo['num_channels'], num_classes=trainInfo['num_classes'])
    
    # 获取全局模型参数并加载
    resp = requests.get(f'http://localhost:{args.port}/newGlobalModel')
    globalStateDict = hexToStateDict(resp.json()['global_model_hex'])
    model.load_state_dict(globalStateDict)
    return model

# 上传本地模型权重到服务器
def uploadWeights(net_glob):
    local_model_hex = stateDictToHex(net_glob.state_dict())
    requests.post(f'http://localhost:{args.port}/newLocalModel/{args.address}', json={
        "local_model_hex": local_model_hex
    })

if __name__ == '__main__':
    # 设备选择 (GPU 或 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 加载数据集
    if args.dataset == "mnist":
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)

        # 根据参数选择 iid 或 non-iid 分割
        if args.iid:
            dataset_train = iid(dataset_train, args.start_train_index, args.end_train_index)
            dataset_test = iid(dataset_test, args.start_test_index, args.end_test_index)
        else:
            dataset_train = non_iid(dataset_train, args.user_id, args.num_users)
            dataset_test = non_iid(dataset_test, args.user_id, args.num_users)
    
    elif args.dataset == "cifar":
        if not args.iid:
            exit("CIFAR 不支持 non-iid 模式")
        
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)

    else:
        exit(f"{args.dataset} 数据集不支持")

    # 初始化节点
    node = Node(device=device, epoch=args.epoch, dataset_train=dataset_train, dataset_test=dataset_test, lr=args.lr)
    
    # 注册节点到服务器
    if not register(len(dataset_train.idxs)):
        print("Failed to register the node!")
        exit()

    # 按照设定的 epoch 数训练
    for epoch in range(args.epoch):
        print(f"Running Epoch {epoch + 1}/{args.epoch}")
        
        # 获取全局模型
        net_glob = getNetGlob().to(device=device)
        
        # 训练本地模型
        node.train(net_glob)
        net_glob.eval()

        # 设置批处理大小并测试模型
        
        batch_size = 64  # 设置批处理大小
        acc_test, loss_test = test_img(net_glob, dataset_test, batch_size, args.gpu)

        # acc_test, loss_test = test_img(net_glob, dataset_test, epoch, args.gpu, batch_size)
        print(f"Test set: Average loss: {loss_test:.4f}, Accuracy: {acc_test * 100:.2f}%")
        
        # 上传本地训练后的权重到服务器
        uploadWeights(net_glob)
        
        print(f"Epoch {epoch + 1}/{args.epoch} completed.\n")
        
        # 等待服务器更新全局模型
        time.sleep(5)

    print("Training completed!")