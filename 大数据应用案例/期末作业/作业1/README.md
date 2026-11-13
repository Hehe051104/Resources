# 大数据应用案例期末项目 - 华盛顿州电动汽车注册数据分析

## 项目概述

基于Hadoop生态系统对华盛顿州电动汽车注册数据（约28.7万条）进行大数据分析，完成10个统计指标分析、Flink流处理模拟、MapReduce与Spark双框架对比、数据可视化展示及市场趋势预测。

## 运行环境

| 组件 | 版本 |
|------|------|
| Python | 3.10+ |
| pandas | 2.0+ |
| numpy | 1.24+ |
| matplotlib | 3.7+ |
| seaborn | 0.12+ |
| scikit-learn | 1.3+ |
| python-pptx | 0.6+（可选，用于生成PPT） |

## 项目文件结构

```
作业1/
├── Electric_Vehicle_Population_Data.csv   # 原始数据集
├── analysis.py                            # 主分析程序
├── generate_ppt.py                        # PPT生成脚本
├── README.md                              # 本文件
├── 大数据应用案例期末项目报告.md              # 实验报告（Markdown）
├── 大数据应用案例期末项目报告.pdf              # 实验报告（PDF）
└── outputs/                               # 输出结果目录
    ├── eda_*.png                          # EDA可视化图表
    ├── fig1_*~fig10_*.png                 # 10个分析任务图表
    ├── fig_extended_*.png                 # 扩展分析图表
    ├── fig_flink_*.png                    # Flink模拟图表
    ├── fig_dual_framework_*.png           # 双框架对比图表
    ├── task*_.csv                         # 各任务统计结果
    └── *.csv                              # 其他统计输出
```

## 运行步骤

```bash
# 1. 安装依赖
pip install pandas numpy matplotlib seaborn scikit-learn python-pptx

# 2. 运行主分析程序（生成所有图表和CSV结果）
python analysis.py

# 3. 生成PPT（可选）
python generate_ppt.py

# 4. 查看结果
# 所有输出保存在 outputs/ 目录下
```

## 统计指标

1. 不同城市新能源汽车数量排名
2. 不同品牌的车辆数量 Top10
3. 不同年份的车辆数量分布
4. 纯电动车与插电式混合动力车数量对比
5. 不同品牌续航里程分布
6. 充电设施与EV注册量关联分析
7. CAFV政策覆盖分析
8. 价格与续航关系分析
9. 车型多样性排名
10. BEV/PHEV比例趋势分析

## 额外加分项

- Flink实时流处理模拟（+3分）
- MapReduce vs Spark双框架对比（+2分）
- 数据可视化展示（+2分，共17张图表）
- 扩展分析：市场趋势预测（+2分）

## 小组成员

| 姓名 | 负责内容 |
|------|----------|
| 崔鹏宇 | Hadoop集群搭建、核心计算程序 |
| 何润彬 | 数据预处理与统计分析 |
| 刘雨涵 | 报告撰写与可视化设计 |
