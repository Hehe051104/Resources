import random,numpy as np,pandas as pd
import torch
from matplotlib import pyplot as plt
import torchvision
from torch.utils.data import DataLoader, random_split, TensorDataset  # 用于数据加载和划分
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # 用于计算回归指标
from torchinfo import summary  # 用于打印模型参数量和计算量


# ==== 配置 ====
use_amp = False                 # 是否开启混合精度 (Automatic Mixed Precision) → 降显存/提速, 大模型用; 小模型关掉更稳
use_compile = False             # PyTorch2 的 torch.compile → JIT 编译优化计算图, 提升速度; 首次编译会卡; 有兼容性问题时关掉
use_aug = False                  # 是否开启数据增强 (RandomCrop/Flip) → 提升泛化; 纯测试/非图像任务设 False
num_workers = 0                 # DataLoader 用多少 CPU 核加载数据; Windows 推荐设 0; Linux 可设为CPU核数的一半
pin_memory = True               # 是否锁页内存 (固定内存) → 加快 CPU → GPU 拷贝速度; 有 CUDA 时一般设 True
max_grad_norm = 0               # 梯度裁剪阈值 (防止梯度爆炸) → 0 表示不裁剪; 常用范围 0.5~5.0
patience = 5                    # 早停耐心值 → 验证集 loss 连续多少轮不下降, 就提前停止
best_path = "房价预测.pt"  # 保存最佳模型参数的路径

def set_seed(seed=42):
    random.seed(seed)                  # 固定 Python 的随机数种子
    np.random.seed(seed)               # 固定 numpy 的随机数种子
    torch.manual_seed(seed)            # 固定 CPU 上 torch 的随机数
    torch.cuda.manual_seed_all(seed)   # 固定所有 GPU 上 torch 的随机数
    torch.backends.cudnn.deterministic = True  # 让 cudnn 算子确定性 (结果可复现, 但可能变慢)
    torch.backends.cudnn.benchmark = False     # 禁用自动选择最快算法; 若输入尺寸固定可设 True 加速
set_seed(42)

# ==== 模型 ====
class Model(torch.nn.Module):   # 定义模型, 继承自 nn.Module
    def __init__(self):
        super(Model, self).__init__()
        self.model = torch.nn.Sequential(  # 顺序容器, 按顺序堆叠层
            torch.nn.Linear(13, 64),      # 输入层 (13维) → 隐藏层1 (64维)
            torch.nn.SiLU(),

            torch.nn.Linear(64, 64),     # 隐藏层1 (64维) → 隐藏层2 (64维)
            torch.nn.SiLU(),

            torch.nn.Linear(64, 64),
            torch.nn.SiLU(),

            torch.nn.Linear(64, 1)       # 隐藏层2 (64维) → 输出层 (1维)
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

    df=pd.read_csv('house prices(1).csv', encoding='gbk')
    X=torch.tensor(df.drop(columns=['MEDV同类房屋价格的中位数']).values,dtype=torch.float32)
    # 对X作标准化
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True)
    X = (X - mean) / std
    y=torch.tensor(df['MEDV同类房屋价格的中位数'].values,dtype=torch.float32).view(-1,1)

    dataset = TensorDataset(X, y)  #Dataset 提供“索引式存储” 不会把 X 和 y 合并成一个张量，而是按索引绑定
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size,test_size])  # 80%训练, 20%测试

    #DataLoader 提供“高效加载 + 批处理 + 并行”
    train_loader=DataLoader(train_set, batch_size=64, shuffle=True,    # DataLoader: 每批64, 打乱顺序
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader=DataLoader(test_set, batch_size=64, shuffle=False,     # 测试集不用 shuffle
                           num_workers=num_workers, pin_memory=pin_memory)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择 GPU 或 CPU

    model=Model().to(device)   # 把模型放到设备上 (GPU/CPU)

    # 使用 torchinfo 输出模型参数量与 FLOPs
    info = summary(
        model,
        input_size=(1, 13),                # 输入形状：batch=1, 特征=13  单样本输入，特征维度13
        dtypes=[torch.float32],
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        verbose=0
    )
    print(info)
    total_macs = info.total_mult_adds
    print(f"≈ FLOPs: {total_macs * 2:,}")


    if use_compile and hasattr(torch, "compile"):   # 若 use_compile=True 且 torch 有 compile 功能
            model = torch.compile(model)                # 用 JIT 编译加速

    loss_fn = torch.nn.MSELoss().to(device)      # 损失函数: 具体什么需要看任务需求
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)  # AdamW 优化器 (自适应学习率+权重衰减)
    epoch = 200                  # 总训练轮数
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,                # 峰值学习率（推荐比初始 lr 高 2~5 倍）
        epochs=epoch,              # 总训练轮数
        steps_per_epoch=len(train_loader), # 每轮训练多少步 (通常 = 训练集样本数 / batch_size)
        pct_start=0.3,              # 前 30% 热身上升学习率，后 70% 逐渐下降
        anneal_strategy='cos'       # 余弦退火，衰减更平滑
    ) # OneCycleLR 学习率调度器  是在每一轮最后用的
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=='cuda' and use_amp)) # AMP混合精度缩放器, 用于防止梯度下溢

    train_loss_list, test_loss_list = [], []       # 存储每轮的训练loss/测试loss
    best_val = float('inf')     # 最佳验证loss
    bad_epochs = 0        # 早停计数器
    total_train_step = 0                 # 记录训练的 step 次数 (每 batch 增加1)

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
            optimizer.zero_grad(set_to_none=True) # 梯度清零 (set_to_none=True 更快更省内存) True用于大模型内存紧张时
            scaler.scale(loss).backward()         # 反向传播 (梯度缩放, 避免溢出/下溢)
            if max_grad_norm:                     # 若设置梯度裁剪
                scaler.unscale_(optimizer)        # 先反缩放梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # 裁剪梯度范数
            scaler.step(optimizer)                # 更新参数
            scaler.update()                       # 更新缩放因子

            total_train_step += 1                 # 累加 step
            if total_train_step % 10 == 0:       # 每100步打印一次loss
                print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))

            train_loss += loss.item()             # 累加loss
        avg_train_loss = train_loss/len(train_loader)   # 平均训练loss
        train_loss_list.append(avg_train_loss) # 平均训练loss

        train_end = time.time()                   # 训练结束计时
        print("本轮训练耗时: {:.10f} 秒".format(train_end - train_start))
        print("-------第 {} 轮训练结束-------".format(i+1))

        print("-------第 {} 轮测试开始-------".format(i+1))
        test_start = time.time()                  # 测试开始计时

        total_loss=0.0
        model.eval()                              # 评估模式 (关闭dropout, BN用滑动均值)
        all_preds, all_trues = [], []

        with torch.no_grad():                     # 不计算梯度 (节省显存)
            for inputs,labels in test_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.autocast("cuda", enabled=(device.type=='cuda' and use_amp)):
                    outs=model(inputs)
                    loss=loss_fn(outs, labels)
                    preds = outs.squeeze(1).cpu().numpy()
                    trues = labels.squeeze(1).cpu().numpy()
                    all_preds.append(preds)
                    all_trues.append(trues)

                total_loss += loss.item()         # 累加测试loss

        y_pred_test = np.concatenate(all_preds)  # 将所有批次的预测结果拼接成一个数组,本来因为batch所以有多维
        y_true_test = np.concatenate(all_trues)
        avg_test_loss = total_loss/len(test_loader)   # 平均测试loss
        test_loss_list.append(avg_test_loss)
        print("整体测试集上的平均Loss: {:.4f}".format(avg_test_loss))

        mae  = mean_absolute_error(y_true_test, y_pred_test)
        mse = mean_squared_error(y_true_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_true_test, y_pred_test)
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

        test_end = time.time()                   # 测试结束计时
        print("本轮测试耗时: {:.10f} 秒".format(test_end - test_start))
        print("-------第 {} 轮测试结束-------".format(i+1))

        # 保存最佳模型
        if avg_test_loss < best_val-1e-6:        # 如果测试loss更小
            best_val = avg_test_loss
            bad_epochs=0
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
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 真实值 vs 预测值对比图
    # 加载最佳模型
    best_model = Model().to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device))  #map_location=device 保证原模型在 GPU 训练
    best_model.eval()

    # 在所有数据上预测
    with torch.no_grad():
        y_pred = best_model(X.to(device)).squeeze(1).cpu().numpy()  #squeeze(1)：去掉预测结果多余的维度 .cpu().numpy()：将预测值转回 CPU，并转换成 NumPy 数组方便后续处理
    y_true = y.squeeze(1).cpu().numpy()

    # 按真实值排序后画折线图  根据排序索引重排真实值和预测值
    order = np.argsort(y_true)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]

    plt.figure()
    plt.plot(y_true_sorted, label='True (sorted)')
    plt.plot(y_pred_sorted, label='Predicted', alpha=0.8)   #alpha=0.8 让预测曲线稍微透明一点，更容易区分
    plt.xlabel('Sample (sorted by true value)')
    plt.ylabel('Target')
    plt.title('True value vs predicted value')
    plt.legend()      #显示图例，标注两条曲线分别是什么
    plt.tight_layout()     #tight_layout() 自动调整子图间距，防止标签被遮挡
    plt.show()


if __name__ == "__main__":
    main()
