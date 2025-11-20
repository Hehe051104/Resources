import random,numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import torchvision
from torch.utils.data import DataLoader
import os, time

# ==== 配置 ====
use_amp = False                 # 是否开启混合精度 (Automatic Mixed Precision) → 降显存/提速, 大模型用; 小模型关掉更稳
use_compile = False             # PyTorch2 的 torch.compile → JIT 编译优化计算图, 提升速度; 首次编译会卡; 有兼容性问题时关掉
use_aug = True                  # 是否开启数据增强 (RandomCrop/Flip) → 提升泛化; 纯测试/非图像任务设 False
num_workers = 0                 # DataLoader 用多少 CPU 核加载数据; Windows 推荐设 0; Linux 可设为CPU核数的一半
pin_memory = True               # 是否锁页内存 (固定内存) → 加快 CPU → GPU 拷贝速度; 有 CUDA 时一般设 True
max_grad_norm = 0               # 梯度裁剪阈值 (防止梯度爆炸) → 0 表示不裁剪; 常用范围 0.5~5.0
patience = 5                    # 早停耐心值 → 验证集 loss 连续多少轮不下降, 就提前停止
best_path = "./best.pt"         # 保存最佳模型参数的路径

def set_seed(seed=42):
    random.seed(seed)                  # 固定 Python 的随机数种子
    np.random.seed(seed)               # 固定 numpy 的随机数种子
    torch.manual_seed(seed)            # 固定 CPU 上 torch 的随机数
    torch.cuda.manual_seed_all(seed)   # 固定所有 GPU 上 torch 的随机数
    torch.backends.cudnn.deterministic = True  # 让 cudnn 算子确定性 (结果可复现, 但可能变慢)
    torch.backends.cudnn.benchmark = False     # 禁用自动选择最快算法; 若输入尺寸固定可设 True 加速
set_seed(42)

# ==== 模型 ====
class Model(nn.Module):   # 定义模型, 继承自 nn.Module
    def __init__(self):
        super(Model, self).__init__()
        self.model = torch.nn.Sequential(  # 顺序容器, 按顺序堆叠层
            torch.nn.Conv2d(3, 16, kernel_size=3, padding='same'),  # 卷积层: 输入通道3(RGB), 输出16, 卷积核3x3, padding=same 保持尺寸
            torch.nn.ReLU(),                                       # 激活函数 ReLU
            torch.nn.BatchNorm2d(16),                              # 批归一化, 稳定训练
            torch.nn.MaxPool2d(2,2),                               # 最大池化, 2x2, 步长2 (下采样一半)

            torch.nn.Conv2d(16, 32, kernel_size=3, padding='same'),# 第二个卷积层: 输入16, 输出32
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2,2),                               # 再次池化, 尺寸再减半

            torch.nn.Flatten(),                                    # 拉平成一维向量
            torch.nn.Linear(32*8*8, 128),                          # 全连接层: 输入特征 32*8*8, 输出128
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),                                 # Dropout, 随机丢弃50%, 防过拟合
            torch.nn.Linear(128, 10)                               # 输出层: 10 类 (CIFAR-10)
        )

    def forward(self, x):
        return self.model(x)  # 前向传播, 直接用 Sequential

# ==== 主训练逻辑 ====
def main():
    # --- 数据 ---
    if use_aug:
        train_tf = torchvision.transforms.Compose([    # 训练集增强: 多个变换组合
            torchvision.transforms.RandomCrop(32, padding=4),   # 随机裁剪 (保持32x32, 周围填充4像素)
            torchvision.transforms.RandomHorizontalFlip(),      # 随机水平翻转
            torchvision.transforms.ToTensor()                   # 转成 Tensor, 并归一化到 [0,1]
        ])
    else:
        train_tf = torchvision.transforms.ToTensor()
    test_tf = torchvision.transforms.ToTensor()       # 测试集只做 ToTensor, 不做增强

    train_set=torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tf) # CIFAR10 训练集
    test_set=torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)  # CIFAR10 测试集

    train_loader=DataLoader(train_set, batch_size=64, shuffle=True,    # DataLoader: 每批64, 打乱顺序
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader=DataLoader(test_set, batch_size=64, shuffle=False,     # 测试集不用 shuffle
                           num_workers=num_workers, pin_memory=pin_memory)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择 GPU 或 CPU

    model=Model().to(device)   # 把模型放到设备上 (GPU/CPU)

    if use_compile and hasattr(torch, "compile"):   # 若 use_compile=True 且 torch 有 compile 功能
        model = torch.compile(model)                # 用 JIT 编译加速

    loss_fn = nn.CrossEntropyLoss().to(device)      # 损失函数: 交叉熵, 多分类常用
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.001)  # AdamW 优化器 (自适应学习率+权重衰减)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)     # 学习率调度器: 余弦退火, T_max=10 表示10个epoch衰减到接近0
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=='cuda' and use_amp)) # AMP混合精度缩放器, 用于防止梯度下溢

    epoch = 10                                     # 总训练轮数
    train_loss_list, test_loss_list = [], []       # 存储每轮的训练loss/测试loss
    best_val = float('inf'); bad_epochs = 0        # 最佳验证loss, 以及早停计数器
    total_train_step = 0                           # 记录训练的 step 次数 (每 batch 增加1)

    for i in range(epoch):
        print("-------第 {} 轮训练开始-------".format(i+1))
        train_start = time.time()                  # 训练开始计时

        train_loss = 0.0
        model.train()                              # 训练模式 (启用dropout, BN可更新)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True) # 把数据放到GPU, non_blocking=True加速拷贝

            with torch.autocast("cuda", enabled=(device.type=='cuda' and use_amp)): # 自动混合精度 (只在GPU时启用)
                outs=model(inputs)                # 前向传播
                loss=loss_fn(outs, labels)        # 计算loss
            optimizer.zero_grad(set_to_none=True) # 梯度清零 (set_to_none=True 更快更省内存)
            scaler.scale(loss).backward()         # 反向传播 (梯度缩放, 避免溢出/下溢)
            if max_grad_norm:                     # 若设置梯度裁剪
                scaler.unscale_(optimizer)        # 先反缩放梯度
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # 裁剪梯度范数
            scaler.step(optimizer)                # 更新参数
            scaler.update()                       # 更新缩放因子

            total_train_step += 1                 # 累加 step
            if total_train_step % 100 == 0:       # 每100步打印一次loss
                print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))

            train_loss += loss.item()             # 累加loss
        train_loss_list.append(train_loss/len(train_loader)) # 平均训练loss

        train_end = time.time()                   # 训练结束计时
        print("本轮训练耗时: {:.2f} 秒".format(train_end - train_start))
        print("-------第 {} 轮训练结束-------".format(i+1))

        print("-------第 {} 轮测试开始-------".format(i+1))
        test_start = time.time()                  # 测试开始计时

        total_loss,total_correct=0.0,0
        model.eval()                              # 评估模式 (关闭dropout, BN用滑动均值)
        with torch.no_grad():                     # 不计算梯度 (节省显存)
            for inputs,labels in test_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.autocast("cuda", enabled=(device.type=='cuda' and use_amp)):
                    outs=model(inputs)
                    loss=loss_fn(outs, labels)
                total_loss += loss.item()         # 累加测试loss
                total_correct += (torch.argmax(outs, 1)==labels).sum().item() # 预测正确数
        avg_test_loss = total_loss/len(test_loader)   # 平均测试loss
        test_loss_list.append(avg_test_loss)
        acc = total_correct/len(test_set)             # 测试集准确率
        print("整体测试集上的Loss: {:.4f}, Acc: {:.4f}".format(avg_test_loss, acc))

        test_end = time.time()                   # 测试结束计时
        print("本轮测试耗时: {:.2f} 秒".format(test_end - test_start))
        print("-------第 {} 轮测试结束-------".format(i+1))

        # 保存最佳模型
        if avg_test_loss < best_val-1e-6:        # 如果测试loss更小
            best_val = avg_test_loss; bad_epochs=0
            torch.save(model.state_dict(), best_path) # 保存当前权重
            print("✔ 已保存最佳模型")
        else:
            bad_epochs+=1
            if bad_epochs>=patience:             # 如果连续patience次没提升 → 早停
                print("⛳ 早停触发"); break

        scheduler.step()                         # 调度器更新学习率

    # 最终可视化 loss 曲线
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(test_loss_list, label='Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.show()

if __name__ == "__main__":
    main()
