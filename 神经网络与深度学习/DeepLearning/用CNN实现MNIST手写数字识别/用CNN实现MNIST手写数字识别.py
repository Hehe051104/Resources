import random , numpy as np
import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
import time , os
from matplotlib import pyplot as plt
from torchinfo import summary
from sklearn.metrics import accuracy_score , precision_recall_fscore_support , confusion_matrix
import seaborn as sns


def set_seed(seed=42):
    random.seed(seed)                  # 固定 Python 的随机数种子
    np.random.seed(seed)               # 固定 numpy 的随机数种子
    torch.manual_seed(seed)            # 固定 CPU 上 torch 的随机数
    torch.cuda.manual_seed_all(seed)   # 固定所有 GPU 上 torch 的随机数
    torch.backends.cudnn.deterministic = True  # 让 cudnn 算子确定性 (结果可复现, 但可能变慢)
    torch.backends.cudnn.benchmark = False     # 禁用自动选择最快算法; 若输入尺寸固定可设 True 加速
set_seed(42)


use_aug = True
best_CNN = './best_CNN.pt'
best_MLP = './best_MLP.pt'
best_CNN_dilated = './best_CNN_dilated.pt'

if use_aug:
    train_tf = torchvision.transforms.Compose([    # 训练集增强: 多个变换组合
        torchvision.transforms.RandomCrop(28, padding=4),   # 随机裁剪
        torchvision.transforms.RandomAffine(degrees=10, translate=(0.1,0.1)), # 随机仿射变换
        torchvision.transforms.ToTensor()                   # 转成 Tensor, 并归一化到 [0,1]
    ])
else:
    train_tf = torchvision.transforms.ToTensor()
test_tf = torchvision.transforms.ToTensor()

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_tf)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_tf)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, pin_memory=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):   # 定义模型, 继承自 nn.Module
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(  # 顺序容器, 按顺序堆叠层
            torch.nn.Conv2d(1, 16, kernel_size=3, padding='same'),  # 卷积层: 输入通道1(灰度), 输出16, 卷积核3x3, padding=same 保持尺寸
            torch.nn.ReLU(),                                       # 激活函数 ReLU
            torch.nn.BatchNorm2d(16),                              # 批归一化, 稳定训练
            torch.nn.MaxPool2d(2,2),                               # 最大池化, 2x2, 步长2 (下采样一半)

            torch.nn.Conv2d(16, 32, kernel_size=3, padding='same'),# 第二个卷积层: 输入16, 输出32
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2,2),                               # 再次池化, 尺寸再减半

            torch.nn.Flatten(),                                    # 拉平成一维向量
            torch.nn.Linear(32*7*7, 128),                          # 全连接层: 输入特征 32*7*7, 输出128
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),                                 # Dropout, 随机丢弃50%, 防过拟合
            torch.nn.Linear(128, 10)                               # 输出层: 10 类 (MNIST)
        )
    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28*28*1 , 256),
            torch.nn.ReLU(),

            torch.nn.Linear(256 , 128),
            torch.nn.ReLU(),

            torch.nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.model(x)

# 空洞卷积模型 Dilated CNN
class DilatedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=2, dilation=2),  # 空洞卷积，dilation=2
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(16, 32, kernel_size=3, padding=2, dilation=2), # 空洞卷积，dilation=2
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Flatten(),
            torch.nn.Linear(32*7*7, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.model(x)

# 可视化第一层卷积特征图
def visualize_feature_maps(model, test_set, device):
    feature_maps = []
    def hook_fn(module, input, output):
        feature_maps.append(output.cpu().detach())
    # 注册钩子到第一层卷积
    hook = model.model[0].register_forward_hook(hook_fn)
    img, _ = test_set[0]
    img = img.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        _ = model(img)
    # 可视化前16个特征图
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 8, figsize=(15, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(feature_maps[0][0, i], cmap='gray')
        ax.axis('off')
    plt.suptitle('第一层卷积特征图')
    plt.show()
    hook.remove()

def main(Model , best_path, epochs , bad_patience ,learning_rate , weight_decay , t_max):
    model = Model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    # SGD + Momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    info = summary(
        model,
        input_size=(1, 1 ,28 ,28),
        dtypes=[torch.float32],
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        verbose=0
    )
    print(info)
    total_macs = info.total_mult_adds
    print(f"≈ FLOPs: {total_macs * 2:,}")

    train_loss_list = []
    test_loss_list = []
    total_train_step = 0
    best_val = float('inf')
    bad_epochs = 0

    for epoch in range(epochs):

        print(f'---------第{epoch+1}轮训练开始---------')
        train_start_time = time.time()

        train_loss = 0
        model.train()

        for inputs, labels in train_loader:
            inputs,labels = inputs.to(device , non_blocking = True), labels.to(device , non_blocking = True)

            outs = model(inputs)
            loss = loss_fn(outs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total_train_step += 1
            if total_train_step % 100 == 0:
                print(f'训练次数: {total_train_step}, Loss: {loss.item():.5f}')

        train_loss_list.append(train_loss/len(train_loader))

        train_finish_time = time.time()
        print(f'第{epoch+1}轮训练时间: {train_finish_time - train_start_time:.4f}秒')
        print(f'---------第{epoch+1}轮训练结束---------')

        scheduler.step()

        print(f'---------第{epoch+1}轮测试开始---------')
        test_start_time = time.time()

        test_loss = 0
        all_pred = []
        all_true = []

        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device , non_blocking = True), labels.to(device , non_blocking = True)

                outs = model(inputs)
                loss = loss_fn(outs, labels)
                test_loss += loss.item()
                preds = torch.argmax(outs, dim=1).cpu().numpy()
                all_pred.extend(preds)
                trues = labels.cpu().numpy()
                all_true.extend(trues)

        avg_test_loss = test_loss/len(test_loader)
        test_loss_list.append(avg_test_loss)

        accuracy = accuracy_score(all_true, all_pred)
        print(f'测试集上平均Loss: {avg_test_loss:.5f}, 准确率: {accuracy:.5f}')

        precision, recall, f1, support = precision_recall_fscore_support(all_true, all_pred, average='macro')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, Support: {support}')


        test_finish_time = time.time()
        print(f'第{epoch+1}轮测试时间: {test_finish_time - test_start_time:.4f}秒')
        print(f'---------第{epoch+1}轮测试结束---------')

        if avg_test_loss < best_val and best_val - avg_test_loss > 1e-3:
            best_val = avg_test_loss
            bad_epochs = 0
            torch.save(model.state_dict() , best_path)
            print(f'验证集Loss降低, 保存当前最佳模型参数到 {best_path}')
        else:
            bad_epochs += 1
            print(f'验证集Loss未降低, 早停计数: {bad_epochs}/{bad_patience}')
            if bad_epochs >= bad_patience:
                print('连续5轮验证集Loss未降低, 提前停止训练')
                break


    #混淆矩阵可视化
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title('Confusion Matrix')
    plt.show()

    model = CNN().to(device)
    model.load_state_dict(torch.load(best_CNN, map_location=device))
    visualize_feature_maps(model, test_set, device)

    # 可视化 训练和测试损失曲线
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(test_loss_list, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    print('CNN模型训练:')
    main(Model = CNN , best_path = best_CNN , epochs = 30 , bad_patience = 5 , learning_rate = 1e-3 , weight_decay = 1e-4 , t_max = 30)
    # print('MLP模型训练:')
    # main(Model = MLP , best_path = best_MLP , epochs = 50 , bad_patience = 5 , learning_rate = 5e-4 , weight_decay = 5e-4 , t_max = 50)
    # print('Dilated CNN模型训练:')
    # main(Model = DilatedCNN , best_path = best_CNN_dilated , epochs = 30 , bad_patience = 5 , learning_rate = 1e-3 , weight_decay = 1e-4 , t_max = 30)