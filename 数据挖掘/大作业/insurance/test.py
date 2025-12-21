# -*- coding: utf-8 -*-
"""
优化版欺诈检测模型 - 修复字段缺失问题
"""
import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

# --- 自动定位目录 ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- 导入模型库 ---
try:
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
except ImportError:
    raise ImportError("请先安装库: pip install lightgbm catboost")

warnings.filterwarnings('ignore')

# 读取数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


def feature_engineering(data):
    """
    统一的特征工程函数 - 已移除导致报错的不确定列
    """
    # 1. 拆分 policy_csl (如果存在)
    if 'policy_csl' in data.columns:
        data[['policy_csl_ll', 'policy_csl_ul']] = data['policy_csl'].str.split('/', n=1, expand=True).astype(int)
        data = data.drop(['policy_csl'], axis=1)

    # 2. 日期处理
    # 转换日期列
    data['policy_bind_date'] = pd.to_datetime(data['policy_bind_date'])
    data['incident_date'] = pd.to_datetime(data['incident_date'])

    # 提取年月日特征
    data['policy_bind_year'] = data['policy_bind_date'].dt.year
    data['policy_bind_month'] = data['policy_bind_date'].dt.month
    data['incident_year'] = data['incident_date'].dt.year
    data['incident_month'] = data['incident_date'].dt.month
    data['incident_day'] = data['incident_date'].dt.day

    # 3. 业务逻辑特征
    # 车龄
    data['auto_year'] = data['auto_year'].astype(int)
    data['car_age'] = data['incident_year'] - data['auto_year']

    # 事故发生距离保单绑定的天数
    data['incident_days'] = (data['incident_date'] - data['policy_bind_date']).dt.days

    # 是否二手车逻辑
    data['is_second_hand'] = (data['policy_bind_year'] > data['auto_year']).astype(int)

    # --- 删除报错的自定义特征 ---
    # data['capital_net'] ... (已移除)
    # data['premium_to_deductible'] ... (已移除，原报错点)

    # 删除原始日期列
    data = data.drop(['policy_bind_date', 'incident_date'], axis=1)

    return data


def preprocessing(train, test):
    # 合并数据进行统一编码，防止编码不一致
    train_len = len(train)
    data = pd.concat([train, test], axis=0, ignore_index=True)

    # 执行特征工程
    data = feature_engineering(data)

    # 区分数值列和类别列
    # 排除不需要的列
    drop_cols = ['policy_id', 'fraud']
    # 自动识别剩下的object列
    cat_cols = [c for c in data.select_dtypes(include=['object']).columns if c not in drop_cols]

    # Label Encoding
    lb = LabelEncoder()
    for col in cat_cols:
        # 转为字符串防止混合类型报错
        data[col] = lb.fit_transform(data[col].astype(str))

    # 分离回训练集和测试集
    train_processed = data.iloc[:train_len].copy()
    test_processed = data.iloc[train_len:].copy()

    # 准备 X 和 Y
    X = train_processed.drop(['policy_id', 'fraud'], axis=1)
    y = train_processed['fraud']
    X_test = test_processed.drop(['policy_id', 'fraud'], axis=1)

    return X, y, X_test, test_processed['policy_id']


print("正在处理数据...")
X, y, X_test, test_ids = preprocessing(train_df, test_df)

# --- 模型定义 ---
# LightGBM 分类器
lgb_params = {
    'num_leaves': 31,
    'max_depth': 7,
    'learning_rate': 0.02,
    'n_estimators': 3000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': True,  # 关键：处理样本不平衡
    'verbose': -1
}

# CatBoost 分类器
cat_params = {
    'depth': 7,
    'learning_rate': 0.02,
    'iterations': 3000,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'auto_class_weights': 'Balanced',  # 关键：处理样本不平衡
    'verbose': 0,
    'allow_writing_files': False
}

# --- 5折交叉验证 (Stratified K-Fold) ---
folds = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2023)

oof_preds = np.zeros(X.shape[0])  # 存储验证集的预测结果
test_preds_lgb = np.zeros(X_test.shape[0])  # 存储测试集的预测结果
test_preds_cat = np.zeros(X_test.shape[0])

print(f"开始 {folds} 折交叉验证训练...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # 1. 训练 LightGBM
    lgb = LGBMClassifier(**lgb_params)
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)],
            callbacks=[])

    # 预测概率 (取类别1的概率)
    val_pred_lgb = lgb.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] += val_pred_lgb * 0.5
    test_preds_lgb += lgb.predict_proba(X_test)[:, 1] / folds

    # 2. 训练 CatBoost
    cat = CatBoostClassifier(**cat_params)
    cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)

    # 预测概率
    val_pred_cat = cat.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] += val_pred_cat * 0.5
    test_preds_cat += cat.predict_proba(X_test)[:, 1] / folds

    # 当前折分数
    fold_auc = roc_auc_score(y_val, (val_pred_lgb + val_pred_cat) / 2)
    print(f"Fold {fold + 1} AUC: {fold_auc:.5f}")

# --- 评估与结果 ---
total_auc = roc_auc_score(y, oof_preds)
print(f"\n=================================")
print(f"整体交叉验证 AUC: {total_auc:.5f}")
print(f"=================================\n")

# 融合两个模型对测试集的预测 (Soft Voting)
final_test_preds = (test_preds_lgb * 0.5) + (test_preds_cat * 0.5)

# 绘制 ROC 曲线
fpr, tpr, thresholds = roc_curve(y, oof_preds)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'CV ROC curve (AUC = {total_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (5-Fold CV)')
plt.legend(loc="lower right")
plt.savefig('roc_curve_optimized.png')
print("ROC曲线已保存为 roc_curve_optimized.png")

# --- 生成提交文件 ---
submit_df = pd.DataFrame()
submit_df['policy_id'] = test_ids
submit_df['fraud'] = final_test_preds

output_file = 'result_optimized.csv'
submit_df.to_csv(output_file, index=False)
print(f"结果文件已生成: {output_file}")