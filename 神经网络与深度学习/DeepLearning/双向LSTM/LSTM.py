from collections import Counter
import re, random, numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --------- 0. 全局配置 ----------
SHOW_PLOTS = True  # True: 保存并直接展示；False: 仅保存
SAVE_DIR = "."

def safe_show():
    """在保存后根据开关选择是否show"""
    if SHOW_PLOTS:
        try:
            plt.show()
        except Exception as e:
            print(f"[WARN] 显示图像失败（可能是无GUI环境）：{e}", file=sys.stderr)
    plt.close()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- 1. 超参 & 设备 ----------
max_vocab   = 20000
max_len     = 200
batch_size  = 128     # 可改为64观察稳定性
epochs      = 10
lr          = 1e-3
weight_decay= 1e-4
best_path   = os.path.join(SAVE_DIR, "best_LSTM.pt")
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2. 读数据 ----------
train_df = pd.read_csv("labeledTrainData.tsv", sep="\t")
test_df  = pd.read_csv("testData.tsv",      sep="\t")
y_np = train_df["sentiment"].to_numpy().astype(np.float32)

X_train_raw = train_df["review"].values
X_test_raw  = test_df["review"].values

# ---------- 3. 文本清理 & 分词（保留否定的 '） ----------
def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)         # 去HTML
    text = re.sub(r"[^a-zA-Z']", " ", text)    # 仅保留字母与单引号
    toks = text.split()
    return [t for t in toks if t != "'"]       # 去掉孤立的 '

X_train_tok = [tokenize(t) for t in X_train_raw]
X_test_tok  = [tokenize(t) for t in X_test_raw]

# ---------- 4. 词表 ----------
counter = Counter([w for sent in X_train_tok for w in sent])
vocab = {"<PAD>": 0, "<UNK>": 1}
for idx, (word, _) in enumerate(counter.most_common(max_vocab), start=2):
    vocab[word] = idx

def text_to_ids(tokens):
    return [vocab.get(w, 1) for w in tokens]   # OOV -> <UNK>=1

# ---------- 5. 索引与pad ----------
X_train_seq = [torch.tensor(text_to_ids(t), dtype=torch.long)[:max_len] for t in X_train_tok]
X_test_seq  = [torch.tensor(text_to_ids(t), dtype=torch.long)[:max_len] for t in X_test_tok]

X_train_pad = pad_sequence(X_train_seq, batch_first=True, padding_value=0)
X_test_pad  = pad_sequence(X_test_seq,  batch_first=True, padding_value=0)
y_train_t   = torch.tensor(train_df["sentiment"].values, dtype=torch.float32)

# ---------- 6. 划分 ----------
idx = np.arange(len(y_np))
tr_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42, stratify=y_np.astype(int))

train_ds = TensorDataset(X_train_pad[tr_idx], y_train_t[tr_idx])
val_ds   = TensorDataset(X_train_pad[val_idx], y_train_t[val_idx])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_pad), batch_size=batch_size*2, shuffle=False)

# ---------- 7. 模型 ----------
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, p_drop=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(hidden_dim*2, 1)

    def forward(self, x):                   # x: [B,T]
        emb = self.embedding(x)             # [B,T,E]
        out, _ = self.lstm(emb)             # [B,T,2H]
        mask = (x != 0).unsqueeze(-1).float()
        out = out * mask
        lengths = mask.sum(dim=1).clamp(min=1.0)
        feat = out.sum(dim=1) / lengths     # 忽略PAD的平均池化
        feat = self.dropout(feat)
        logits = self.fc(feat).squeeze(1)   # [B]
        return logits

model = BiLSTM(vocab_size=len(vocab)).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                       patience=2, verbose=True)

# ---------- 8. 训练 ----------
best_val = float("inf")
bad, patience = 0, 3
best_threshold = 0.5
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1@0.5": []}

for epoch in range(1, epochs+1):
    # Train
    model.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        tr_loss += loss.item()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        tr_correct += (preds == yb).sum().item()
        tr_total += yb.numel()

    train_loss = tr_loss / len(train_loader)
    train_acc  = tr_correct / tr_total

    # Val
    model.eval()
    va_loss, va_correct, va_total = 0.0, 0, 0
    all_labels, all_probs = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            va_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            va_correct += (preds == yb).sum().item()
            va_total += yb.numel()
            all_labels.extend(yb.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss = va_loss / len(val_loader)
    val_acc  = va_correct / va_total
    val_preds_05 = (np.array(all_probs) >= 0.5).astype(int)
    val_f1_05 = f1_score(all_labels, val_preds_05)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1@0.5"].append(val_f1_05)

    print(f"Epoch {epoch:02d} | "
          f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, F1@0.5={val_f1_05:.4f}")

    improved = val_loss < best_val - 1e-3
    if improved:
        best_val = val_loss
        torch.save(model.state_dict(), best_path)
        print(f"↓ 保存最佳模型到 {best_path}")
        bad = 0
        # 在当前验证集上调阈值
        ts = np.linspace(0.3, 0.7, 41)
        best_t, best_f1 = 0.5, -1
        for t in ts:
            pred_t = (np.array(all_probs) >= t).astype(int)
            f1_t = f1_score(all_labels, pred_t)
            if f1_t > best_f1:
                best_f1, best_t = f1_t, t
        best_threshold = float(best_t)
        print(f"* 更新最佳阈值: threshold={best_threshold:.3f}, F1={best_f1:.4f}")
    else:
        bad += 1

    scheduler.step(val_loss)
    if bad >= patience:
        print("Early Stopping.")
        break

# ---------- 8.1 训练曲线：保存并直接展示 ----------
def plot_curves(history):
    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"],   label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Train/Val Loss")
    plt.legend(); plt.tight_layout()
    path = os.path.join(SAVE_DIR, "curve_loss.png")
    plt.savefig(path, dpi=150)
    print(f"✅ 已保存：{path}")
    safe_show()

    # Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"],   label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Train/Val Accuracy")
    plt.legend(); plt.tight_layout()
    path = os.path.join(SAVE_DIR, "curve_acc.png")
    plt.savefig(path, dpi=150)
    print(f"✅ 已保存：{path}")
    safe_show()

plot_curves(history)

# ---------- 9. 验证集报告 & 混淆矩阵（直接展示） ----------
model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()
all_labels, all_preds = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        probs = torch.sigmoid(model(xb)).cpu().numpy()
        all_preds.extend((probs >= best_threshold).astype(int))
        all_labels.extend(yb.numpy())

print("\n=== 分类报告（Validation, BiLSTM）===")
print(classification_report(all_labels, all_preds, digits=4))
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["neg(0)","pos(1)"], yticklabels=["neg(0)","pos(1)"])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (Validation) - BiLSTM")
plt.tight_layout()
path = os.path.join(SAVE_DIR, "cm_bilstm.png")
plt.savefig(path, dpi=150)
print(f"✅ 已保存：{path}")
safe_show()

# ---------- 10. 测试集推理 & 提交 ----------
all_probs = []
with torch.no_grad():
    for (xb,) in test_loader:
        xb = xb.to(device)
        probs = torch.sigmoid(model(xb))
        all_probs.extend(probs.cpu().tolist())

preds = (np.array(all_probs) >= best_threshold).astype(int)
submission = pd.DataFrame({"id": test_df["id"], "sentiment": preds})
sub_path = os.path.join(SAVE_DIR, "submission.csv")
submission.to_csv(sub_path, index=False)
print(f"✅ 已生成 submission.csv（阈值={best_threshold:.3f}） -> {sub_path}")

# ---------- 11. 基线：TF-IDF + 逻辑回归（保存并显示混淆矩阵） ----------
print("\n=== 运行基线模型（TF-IDF + LogisticRegression） ===")
X_tr_text = [X_train_raw[i] for i in tr_idx]
y_tr = y_np[tr_idx].astype(int)
X_val_text = [X_train_raw[i] for i in val_idx]
y_val = y_np[val_idx].astype(int)

tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=2)
X_tr_vec  = tfidf.fit_transform(X_tr_text)
X_val_vec = tfidf.transform(X_val_text)

lr_clf = LogisticRegression(solver="liblinear", max_iter=1000)
lr_clf.fit(X_tr_vec, y_tr)
val_pred_lr = lr_clf.predict(X_val_vec)

print("\n=== 分类报告（Validation, TF-IDF+LR）===")
print(classification_report(y_val, val_pred_lr, digits=4))
cm_lr = confusion_matrix(y_val, val_pred_lr)
plt.figure(figsize=(5,4))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Greens",
            xticklabels=["neg(0)","pos(1)"], yticklabels=["neg(0)","pos(1)"])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (Validation) - TFIDF+LR")
plt.tight_layout()
path = os.path.join(SAVE_DIR, "cm_tfidf_lr.png")
plt.savefig(path, dpi=150)
print(f"✅ 已保存：{path}")
safe_show()

# 验证集结果对比（打印）
f1_lr  = f1_score(y_val, val_pred_lr)
acc_lr = accuracy_score(y_val, val_pred_lr)
f1_bi  = f1_score(all_labels, all_preds)
acc_bi = accuracy_score(all_labels, all_preds)
print("\n=== 验证集结果对比（同一划分）===")
print(f"BiLSTM      -> Acc: {acc_bi:.4f}, F1: {f1_bi:.4f}, threshold={best_threshold:.3f}")
print(f"TFIDF + LR  -> Acc: {acc_lr:.4f}, F1: {f1_lr:.4f}")
print("（注：若无GUI环境导致不弹窗，图已保存到当前目录。）")
