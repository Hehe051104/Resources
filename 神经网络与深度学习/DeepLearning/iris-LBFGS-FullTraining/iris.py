import random, numpy as np, pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==== 配置 ====
use_compile = False
best_path = "./iris预测.pt"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# ==== 模型 ====
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 3)   # logits
        )
    def forward(self, x):
        return self.model(x)

import matplotlib.pyplot as plt
import numpy as np

def plot_scatter_matrix(X, y, class_names, feature_names, save_path=None,alpha=0.85, s=20, bins=16):
    """
    只绘制“散点矩阵 + 对角线直方图”的一张图。
    X: (N, D) 标准化后的训练特征
    y: (N,) 0..K-1 的标签
    class_names: 长度 K 的类别名列表（你的 uniques）
    feature_names: 长度 D 的特征名列表
    """
    X = np.asarray(X)
    y = np.asarray(y)
    D = X.shape[1]
    fig, axes = plt.subplots(D, D, figsize=(11, 11))
    # 颜色/标记：三类够用，更多类可自行扩展
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    classes = np.unique(y)

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if i == j:
                # 对角：各类的直方图
                for k, c in enumerate(classes):
                    ax.hist(X[y==c, j], bins=bins, alpha=0.6, density=False,
                            histtype='stepfilled', label=str(class_names[c]),
                            edgecolor='none')
            else:
                # 非对角：散点
                for k, c in enumerate(classes):
                    ax.scatter(X[y==c, j], X[y==c, i],
                               s=s, alpha=alpha, marker=markers[k % len(markers)],
                               label=str(class_names[c]))
            # 只在最下行/最左列显示坐标刻度标签，避免挤成一团
            if i < D-1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(feature_names[j])
            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(feature_names[i])

    # 合并图例到整幅图顶部
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:len(classes)], labels[:len(classes)],
               loc='upper center', ncol=len(classes), bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.show()


def plot_true_vs_pred(y_true, y_pred, class_names, save_path=None):
    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(y_true, y_pred, s=40, alpha=0.7)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=1)  # 理想对角线
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=30)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("True category"); ax.set_ylabel("Prediction category")
    ax.set_title("(Test set) true value vs predicted value")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.show()
    return fig, ax


def main():
    # --- 读取 & 清洗 ---
    df = pd.read_csv("iris.data", header=None)
    df = df.dropna()                                # 以防最后有空行
    df.columns = ['sepal_len','sepal_wid','petal_len','petal_wid','label']

    # 标签编码：字符串 -> 索引(0/1/2)
    X_np = df[['sepal_len','sepal_wid','petal_len','petal_wid']].values.astype('float32')

    y_np, uniques = pd.factorize(df['label'])   # y_np 是 0/1/2, uniques 是类别名字   标签编码

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
    )

    mean = X_train_np.mean(axis=0, keepdims=True)
    std  = X_train_np.std(axis=0, keepdims=True).clip(min=1e-8)
    X_train_np = (X_train_np - mean) / std
    X_test_np  = (X_test_np  - mean) / std


    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_test  = torch.tensor(X_test_np,  dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    y_test  = torch.tensor(y_test_np,  dtype=torch.long)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test  = X_test.to(device)
    y_test  = y_test.to(device)

    example=np.array([[5.1, 3.5, 1.4, 0.2],[5.7, 3.0, 4.2, 1.2],[6.7, 3.0, 5.2, 2.3]])
    example_norm = (example - mean) / std
    ex=torch.tensor(example_norm, dtype=torch.float32).to(device)

    model = Model().to(device)
    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)  #正则化参数，用于防止模型对标签的过度自信预测
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=50, history_size=10)  #lr=1.0：学习率（通常设1.0） max_iter=50：每次迭代内最大函数评估次数 history_size=10：存储的过去曲率信息数量

    model.train()
    def closure():
        optimizer.zero_grad()
        logits = model(X_train)           # 全量前向
        loss = loss_fn(logits, y_train)
        loss.backward()
        return loss

    # L-BFGS 外圈几次即可（Iris 很快收敛）
    for step in range(10):
        loss = optimizer.step(closure)
        print(f"LBFGS step {step+1} | loss={loss.item():.6f}")


    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_true = y_test.cpu().numpy()
        ex_logits=model(ex)

    acc  = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, average='macro')   # macro: 类别均衡
    rec  = recall_score(y_true, preds, average='macro')
    f1   = f1_score(y_true, preds, average='macro')
    print(f"测试集: acc={acc:.4f} | prec={prec:.4f} | rec={rec:.4f} | f1={f1:.4f}")

    res=torch.argmax(ex_logits,dim=1).cpu().numpy()
    print('随机示例的预测结果为:')
    print(uniques[res].tolist())

    torch.save(model.state_dict(), best_path)
    print('模型已保存至', best_path)


    # ===== 调用：只画训练集；你可以按需切换 orig/std 两套视角 =====
    # 数据切分与标准化完成后（X_train_np / y_train_np 已就绪）
    feature_names = ['sepal_len','sepal_wid','petal_len','petal_wid']

    # 只画这一个图，其他可视化全部注释或删除
    plot_scatter_matrix(
        X_train_np, y_train_np,
        class_names=list(uniques),        # uniques 就是你 factorize 得到的类别名
        feature_names=feature_names,
        save_path="./train_scatter_matrix.png"   # 想只显示不保存就传 None
    )
    print('散点矩阵已保存至 ./train_scatter_matrix.png')

    plot_true_vs_pred(y_true, preds, class_names=list(uniques), save_path="./test_true_vs_pred.png")
    print('测试集 真实值 vs 预测值 图已保存至 ./test_true_vs_pred.png')


if __name__ == "__main__":
    main()
