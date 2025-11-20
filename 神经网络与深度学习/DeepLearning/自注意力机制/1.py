import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import time

# ==================== 1. 数据加载和预处理 ====================
def load_lcqmc_data(filepath):
    """
    加载LCQMC数据集
    文件格式: text1 \t text2 \t label
    """
    data = []
    print(f"正在加载数据: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        # 跳过第一行标题
        next(f)

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                text1, text2, label = parts
                try:
                    data.append((text1, text2, int(label)))
                except ValueError:
                    # 跳过无效行
                    continue

    print(f"✓ 加载完成: {len(data)}条数据")
    return data

class TextPreprocessor:
    """文本预处理器"""
    def __init__(self, vocab_size=5000):
        self.vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
        self.vocab_size = vocab_size
        self.max_len = 64

    def build_vocab(self, data):
        """构建词汇表"""
        print("正在构建词汇表...")
        char_freq = {}

        for text1, text2, _ in data:
            for char in text1 + text2:
                char_freq[char] = char_freq.get(char, 0) + 1

        # 按频率排序，取top vocab_size-4个字符
        sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
        for char, _ in sorted_chars[:self.vocab_size-4]:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        print(f"✓ 词汇表构建完成，大小: {len(self.vocab)}")

    def encode(self, text1, text2):
        """
        编码文本对
        格式: [CLS] text1 [SEP] text2 [PAD]...
        """
        tokens = [self.vocab['[CLS]']]

        # 编码text1（最多30个字符）
        for char in text1[:30]:
            tokens.append(self.vocab.get(char, self.vocab['[UNK]']))

        tokens.append(self.vocab['[SEP]'])

        # 编码text2（最多30个字符）
        for char in text2[:30]:
            tokens.append(self.vocab.get(char, self.vocab['[UNK]']))

        # 填充到max_len
        if len(tokens) < self.max_len:
            tokens += [self.vocab['[PAD]']] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]

        return tokens

class LCQMCDataset(Dataset):
    """LCQMC数据集"""
    def __init__(self, data, preprocessor):
        self.data = data
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text1, text2, label = self.data[idx]
        tokens = self.preprocessor.encode(text1, text2)
        return torch.LongTensor(tokens), torch.FloatTensor([label])

# ==================== 2. 优化的模型 ====================
class OptimizedSelfAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 多头注意力计算
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)

        return self.out(out)

class OptimizedSemanticMatchingModel(nn.Module):
    """优化的语义匹配模型（带残差连接和层归一化）"""
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super().__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(64, embed_size)

        # 多层自注意力（带残差连接和层归一化）
        self.attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': OptimizedSelfAttention(embed_size),
                'norm1': nn.LayerNorm(embed_size),
                'ffn': nn.Sequential(
                    nn.Linear(embed_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, embed_size)
                ),
                'norm2': nn.LayerNorm(embed_size)
            }) for _ in range(num_layers)
        ])

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(embed_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 嵌入 + 位置编码
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)

        # 多层自注意力（带残差连接）
        for layer in self.attention_layers:
            # 自注意力 + 残差 + 层归一化
            attn_out = layer['attn'](x)
            x = layer['norm1'](x + attn_out)

            # 前馈网络 + 残差 + 层归一化
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)

        # 池化（取平均） + 分类
        x = x.mean(dim=1)
        return self.classifier(x)

# ==================== 3. 训练器 ====================
class Trainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self, train_loader, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for inputs, labels in tqdm(train_loader, desc='训练中'):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        return total_loss / len(val_loader), acc

    def train(self, train_loader, val_loader, epochs=10, lr=0.001):
        """完整训练流程"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        print(f"\n开始训练 (设备: {self.device})")
        print(f"总epochs: {epochs}, 学习率: {lr}")
        print("-" * 60)

        start_time = time.time()
        best_acc = 0

        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader, optimizer, criterion)

            # 验证
            val_loss, val_acc = self.evaluate(val_loader, criterion)

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')

            print(f'Epoch {epoch+1}/{epochs}')
            print(f'  训练损失: {train_loss:.4f}')
            print(f'  验证损失: {val_loss:.4f}')
            print(f'  验证准确率: {val_acc:.4f} (最佳: {best_acc:.4f})')
            print("-" * 60)

        training_time = time.time() - start_time
        print(f'\n✓ 训练完成! 总耗时: {training_time:.2f}秒')
        return training_time

# ==================== 4. 模型评估 ====================
def evaluate_model(model, test_loader, device='cuda'):
    """在测试集上评估模型"""
    print("\n开始测试...")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='测试中'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    inference_time = time.time() - start_time

    # 计算评估指标
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\n" + "=" * 60)
    print("测试集评估结果")
    print("=" * 60)
    print(f'准确率 (Accuracy):  {acc:.4f}')
    print(f'精确率 (Precision): {precision:.4f}')
    print(f'召回率 (Recall):    {recall:.4f}')
    print(f'F1分数 (F1 Score):  {f1:.4f}')
    print(f'推理时间:           {inference_time:.2f}秒')
    print("=" * 60)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'inference_time': inference_time
    }

# ==================== 5. 结果可视化 ====================
def plot_results(history, metrics, save_path='results.png'):
    """可视化训练过程和评估结果"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 准确率曲线
    axes[1].plot(history['val_acc'], marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].grid(True, alpha=0.3)

    # 3. 混淆矩阵
    cm = confusion_matrix(metrics['labels'], metrics['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                xticklabels=['不相似', '相似'],
                yticklabels=['不相似', '相似'])
    axes[2].set_xlabel('预测标签')
    axes[2].set_ylabel('真实标签')
    axes[2].set_title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n✓ 可视化结果已保存: {save_path}')
    plt.show()

# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("LCQMC文本语义匹配模型优化实验")
    print("=" * 60)

    # ========== 1. 加载数据 ==========
    print("\n【步骤1】数据加载")
    train_data = load_lcqmc_data('train.tsv')  # 修改为你的实际路径
    dev_data = load_lcqmc_data('dev.tsv')
    test_data = load_lcqmc_data('test.tsv')

    # ========== 2. 构建词汇表 ==========
    print("\n【步骤2】构建词汇表")
    preprocessor = TextPreprocessor(vocab_size=5000)
    preprocessor.build_vocab(train_data)

    # ========== 3. 创建数据集 ==========
    print("\n【步骤3】创建数据加载器")
    train_dataset = LCQMCDataset(train_data, preprocessor)
    dev_dataset = LCQMCDataset(dev_data, preprocessor)
    test_dataset = LCQMCDataset(test_data, preprocessor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    print(f"✓ 训练集: {len(train_dataset)}条")
    print(f"✓ 验证集: {len(dev_dataset)}条")
    print(f"✓ 测试集: {len(test_dataset)}条")

    # ========== 4. 创建模型 ==========
    print("\n【步骤4】创建模型")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OptimizedSemanticMatchingModel(
        vocab_size=len(preprocessor.vocab),
        embed_size=128,
        hidden_size=256,
        num_layers=2
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型参数量: {total_params:,}")
    print(f"✓ 使用设备: {device}")

    # ========== 5. 训练模型 ==========
    print("\n【步骤5】训练模型")
    trainer = Trainer(model, device)
    training_time = trainer.train(train_loader, dev_loader, epochs=10, lr=0.001)

    # ========== 6. 测试模型 ==========
    print("\n【步骤6】测试模型")
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    metrics = evaluate_model(model, test_loader, device)

    # ========== 7. 可视化结果 ==========
    print("\n【步骤7】生成可视化")
    plot_results(trainer.history, metrics)

    # ========== 8. 性能总结 ==========
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    print(f"训练时间:   {training_time:.2f}秒")
    print(f"推理时间:   {metrics['inference_time']:.2f}秒")
    print(f"最终F1分数: {metrics['f1']:.4f}")
    print(f"最终准确率: {metrics['accuracy']:.4f}")
    print("=" * 60)

    print("\n✓ 实验完成!")

if __name__ == "__main__":
    main()