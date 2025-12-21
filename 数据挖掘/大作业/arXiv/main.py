import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt  # <--- 新增: 引入绘图库

# --- 设置绘图风格和字体 ---
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
file_path = 'arxiv-metadata-oai-2019.json'

# 初始化用于存储统计结果的变量
cs_2019_counter = Counter()
author_counter = Counter()
cat_total_counts = Counter()
cat_code_counts = Counter()

def is_paper_in_2019(versions):
    """判断论文是否在2019年提交/更新"""
    if not versions:
        return False
    for v in versions:
        if '2019' in v.get('created', ''):
            return True
    return False

def has_source_code(abstract, comments):
    """判断是否包含源代码"""
    text = (str(abstract) + " " + str(comments)).lower()
    keywords = ['github', 'gitlab', 'bitbucket', 'code available']
    for kw in keywords:
        if kw in text:
            return True
    return False

print("开始处理数据，请稍候...")

# 逐行读取文件
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 获取基本字段
            categories = paper.get('categories', '').split()
            versions = paper.get('versions', [])
            authors_parsed = paper.get('authors_parsed', [])
            abstract = paper.get('abstract', '')
            comments = paper.get('comments', '')

            # --- 任务1: 统计2019年 CS 方向 ---
            if is_paper_in_2019(versions):
                for cat in categories:
                    if cat.startswith('cs.'):
                        cs_2019_counter[cat] += 1

            # --- 任务2: 统计所有论文作者 ---
            for author in authors_parsed:
                first_name = author[1] if len(author) > 1 else ""
                last_name = author[0] if len(author) > 0 else ""
                full_name = f"{first_name} {last_name}".strip()
                if full_name:
                    author_counter[full_name] += 1

            # --- 任务3: 统计所有类别的代码比例 ---
            has_code = has_source_code(abstract, comments)
            for cat in categories:
                cat_total_counts[cat] += 1
                if has_code:
                    cat_code_counts[cat] += 1
except FileNotFoundError:
    print(f"错误: 找不到文件 {file_path}")
    exit()

# === 结果展示与可视化 ===

# 1. 创建画布 (3个子图)
fig, axes = plt.subplots(3, 1, figsize=(10, 18)) # 3行1列
plt.subplots_adjust(hspace=0.4) # 调整子图间距

# --- 任务1 可视化 ---
print("\n=== 任务1: 2019年计算机(CS)各方向论文数量 前10 ===")
df_task1 = pd.DataFrame(cs_2019_counter.items(), columns=['Category', 'Count'])
df_task1 = df_task1.sort_values('Count', ascending=False).reset_index(drop=True)

if not df_task1.empty:
    # 取前15个画图
    top_cs = df_task1.head(15).sort_values('Count', ascending=True) # 反转顺序以便在barh中从上到下显示
    axes[0].barh(top_cs['Category'], top_cs['Count'], color='steelblue')
    axes[0].set_title('2019年 CS方向论文数量 Top 15')
    axes[0].set_xlabel('数量')
    print(df_task1.head(10)) # 打印前10到控制台
else:
    axes[0].text(0.5, 0.5, '无数据', ha='center')

# --- 任务2 可视化 ---
print("\n=== 任务2: 所有论文作者出现频率 前10 ===")
df_task2 = pd.DataFrame(author_counter.items(), columns=['Author Name', 'Paper Count'])
df_task2 = df_task2.sort_values('Paper Count', ascending=False).reset_index(drop=True)

# 取前10个画图
top_authors = df_task2.head(10).sort_values('Paper Count', ascending=True)
axes[1].barh(top_authors['Author Name'], top_authors['Paper Count'], color='salmon')
axes[1].set_title('高产作者 Top 10')
axes[1].set_xlabel('论文数量')
print(df_task2.head(10))

# --- 任务3 可视化 ---
print("\n=== 任务3: 各类别源代码比例 前10 ===")
task3_data = []
for cat, total in cat_total_counts.items():
    code_num = cat_code_counts[cat]
    ratio = code_num / total
    task3_data.append({'Category': cat, 'Total': total, 'With_Code': code_num, 'Ratio': ratio})

df_task3 = pd.DataFrame(task3_data).sort_values(by='Ratio', ascending=False)

# 可视化过滤：只展示总样本数 > 50 的类别，避免 1/1=100% 这种无意义数据霸榜
df_viz3 = df_task3[df_task3['Total'] > 50].head(15).sort_values('Ratio', ascending=True)

axes[2].barh(df_viz3['Category'], df_viz3['Ratio'], color='mediumseagreen')
axes[2].set_title('代码开源率最高的类别 Top 15 (样本数>50)')
axes[2].set_xlabel('比例')
# 设置x轴为百分比格式
axes[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

print(df_task3.head(10))

# 显示所有图表
plt.show()