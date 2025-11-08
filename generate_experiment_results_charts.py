# -*- coding: utf-8 -*-
"""
生成实验结果分析图表
- 角度1：不同预测阈值的性能对比
- 角度2：与XGBoost模型的对比
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无界面环境
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import numpy as np
import os

# 中文字体设置
def setup_chinese_font():
    font_candidates = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Source Han Sans CN']
    available = [f.name for f in fm.fontManager.ttflist]
    chosen = None
    for name in font_candidates:
        if name in available:
            chosen = name
            break
    ch_font = None
    if chosen:
        plt.rcParams['font.sans-serif'] = [chosen]
    else:
        font_path = r"C:\Windows\Fonts\msyh.ttc"
        if os.path.exists(font_path):
            ch_font = FontProperties(fname=font_path)
    plt.rcParams['axes.unicode_minus'] = False
    return ch_font

CH_FONT = setup_chinese_font()

# ==================== 角度1：不同预测阈值的性能对比 ====================
print("生成角度1：不同预测阈值的性能对比图...")

# 数据
thresholds = [0.55, 0.45, 0.35]
f1_scores = [0.069739, 0.074553, 0.072840]
precision_scores = [0.061084, 0.067485, 0.090760]
recall_scores = [0.081254, 0.083275, 0.060830]

# 图1：柱状图+折线图对比
fig1, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(len(thresholds))
width = 0.25

bars1 = ax1.bar(x - width, f1_scores, width, label='F1值', color='#4472C4', alpha=0.8)
bars2 = ax1.bar(x, precision_scores, width, label='精确度(Precision)', color='#70AD47', alpha=0.8)
bars3 = ax1.bar(x + width, recall_scores, width, label='召回率(Recall)', color='#FFC000', alpha=0.8)

# 添加折线图显示趋势
ax1.plot(x - width, f1_scores, marker='o', linewidth=2, markersize=8, color='#2E5090', linestyle='--', alpha=0.7)
ax1.plot(x, precision_scores, marker='s', linewidth=2, markersize=8, color='#4A7C59', linestyle='--', alpha=0.7)
ax1.plot(x + width, recall_scores, marker='^', linewidth=2, markersize=8, color='#CC9900', linestyle='--', alpha=0.7)

ax1.set_xlabel('预测阈值', fontsize=12)
ax1.set_ylabel('指标值', fontsize=12)
ax1.set_title('KMeans-GBDT模型不同预测阈值下的性能对比', fontsize=14, weight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{t:.2f}' for t in thresholds])
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.5f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('threshold_comparison_bar.png', dpi=300, bbox_inches='tight')
print("  已保存: threshold_comparison_bar.png")
plt.close()

# 图2：折线图显示阈值变化趋势
fig2, ax2 = plt.subplots(figsize=(10, 6))
# 反转顺序以便从左到右显示阈值从高到低
thresholds_sorted = sorted(thresholds, reverse=True)
f1_sorted = [f1_scores[thresholds.index(t)] for t in thresholds_sorted]
precision_sorted = [precision_scores[thresholds.index(t)] for t in thresholds_sorted]
recall_sorted = [recall_scores[thresholds.index(t)] for t in thresholds_sorted]

ax2.plot(thresholds_sorted, f1_sorted, marker='o', linewidth=2, markersize=8, label='F1值', color='#4472C4')
ax2.plot(thresholds_sorted, precision_sorted, marker='s', linewidth=2, markersize=8, label='精确度(Precision)', color='#70AD47')
ax2.plot(thresholds_sorted, recall_sorted, marker='^', linewidth=2, markersize=8, label='召回率(Recall)', color='#FFC000')

ax2.set_xlabel('预测阈值', fontsize=12)
ax2.set_ylabel('指标值', fontsize=12)
ax2.set_title('预测阈值对模型性能的影响趋势', fontsize=14, weight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3, linestyle='--')
ax2.set_xlim(0.30, 0.60)

# 添加数值标签
for i, (t, f1, p, r) in enumerate(zip(thresholds_sorted, f1_sorted, precision_sorted, recall_sorted)):
    ax2.text(t, f1, f'{f1:.5f}', ha='center', va='bottom', fontsize=8)
    ax2.text(t, p, f'{p:.5f}', ha='center', va='bottom', fontsize=8)
    ax2.text(t, r, f'{r:.5f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('threshold_comparison_line.png', dpi=300, bbox_inches='tight')
print("  已保存: threshold_comparison_line.png")
plt.close()

# ==================== 角度2：与XGBoost模型的对比（2000万条数据）====================
print("生成角度2：XGBoost与KMeans-GBDT模型性能对比图（2000万条数据）...")

# 数据（2000万条数据）
models_20m = ['XGBoost', 'KMeans-GBDT']
f1_comparison_20m = [0.001083, 0.001513]
precision_comparison_20m = [0.035928, 0.085324]
recall_comparison_20m = [0.000550, 0.000763]

# 图3：柱状图+折线图对比
fig3, ax3 = plt.subplots(figsize=(10, 6))
x = np.arange(len(models_20m))
width = 0.25

bars1 = ax3.bar(x - width, f1_comparison_20m, width, label='F1值', color='#4472C4', alpha=0.8)
bars2 = ax3.bar(x, precision_comparison_20m, width, label='精确度(Precision)', color='#70AD47', alpha=0.8)
bars3 = ax3.bar(x + width, recall_comparison_20m, width, label='召回率(Recall)', color='#FFC000', alpha=0.8)

# 添加折线图显示趋势
ax3.plot(x - width, f1_comparison_20m, marker='o', linewidth=2, markersize=8, color='#2E5090', linestyle='--', alpha=0.7)
ax3.plot(x, precision_comparison_20m, marker='s', linewidth=2, markersize=8, color='#4A7C59', linestyle='--', alpha=0.7)
ax3.plot(x + width, recall_comparison_20m, marker='^', linewidth=2, markersize=8, color='#CC9900', linestyle='--', alpha=0.7)

ax3.set_xlabel('模型', fontsize=12)
ax3.set_ylabel('指标值', fontsize=12)
ax3.set_title('XGBoost与KMeans-GBDT模型性能对比（2000万条数据）', fontsize=14, weight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(models_20m)
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('model_comparison_bar.png', dpi=300, bbox_inches='tight')
print("  已保存: model_comparison_bar.png")
plt.close()

# ==================== 角度3：Baseline与KMeans-GBDT模型对比（0.45阈值）====================
print("生成角度3：Baseline与KMeans-GBDT模型性能对比图（0.45阈值）...")

# 数据（Baseline和KMeans-GBDT在0.45阈值下的全量数据结果）
models_baseline = ['Baseline', 'KMeans-GBDT\n(0.45阈值)']
f1_baseline = [0.005917, 0.074553]
precision_baseline = [0.003062, 0.067485]
recall_baseline = [0.087153, 0.083275]

# 图6：柱状图+折线图对比
fig6, ax6 = plt.subplots(figsize=(10, 6))
x_baseline = np.arange(len(models_baseline))
width = 0.25

bars1_b = ax6.bar(x_baseline - width, f1_baseline, width, label='F1值', color='#4472C4', alpha=0.8)
bars2_b = ax6.bar(x_baseline, precision_baseline, width, label='精确度(Precision)', color='#70AD47', alpha=0.8)
bars3_b = ax6.bar(x_baseline + width, recall_baseline, width, label='召回率(Recall)', color='#FFC000', alpha=0.8)

# 添加折线图显示趋势
ax6.plot(x_baseline - width, f1_baseline, marker='o', linewidth=2, markersize=8, color='#2E5090', linestyle='--', alpha=0.7)
ax6.plot(x_baseline, precision_baseline, marker='s', linewidth=2, markersize=8, color='#4A7C59', linestyle='--', alpha=0.7)
ax6.plot(x_baseline + width, recall_baseline, marker='^', linewidth=2, markersize=8, color='#CC9900', linestyle='--', alpha=0.7)

ax6.set_xlabel('模型', fontsize=12)
ax6.set_ylabel('指标值', fontsize=12)
ax6.set_title('Baseline与KMeans-GBDT模型性能对比（0.45阈值）', fontsize=14, weight='bold')
ax6.set_xticks(x_baseline)
ax6.set_xticklabels(models_baseline)
ax6.legend(fontsize=10)
ax6.grid(axis='y', alpha=0.3, linestyle='--')

# 添加数值标签
for bars in [bars1_b, bars2_b, bars3_b]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.5f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('baseline_kmeans_gbdt_comparison_bar.png', dpi=300, bbox_inches='tight')
print("  已保存: baseline_kmeans_gbdt_comparison_bar.png")
plt.close()

# 图4：雷达图对比（虽然只有3个指标，但可以展示）
fig4 = plt.figure(figsize=(10, 8))
ax4 = fig4.add_subplot(111, projection='polar')

# 指标名称
categories = ['F1值', '精确度\n(Precision)', '召回率\n(Recall)']
N = len(categories)

# 计算角度
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # 闭合

# 数据归一化（为了在同一尺度上比较）
max_values = [max(f1_comparison_20m), max(precision_comparison_20m), max(recall_comparison_20m)]
xgb_normalized = [f1_comparison_20m[0]/max_values[0], precision_comparison_20m[0]/max_values[1], recall_comparison_20m[0]/max_values[2]]
kmeans_gbdt_normalized = [f1_comparison_20m[1]/max_values[0], precision_comparison_20m[1]/max_values[1], recall_comparison_20m[1]/max_values[2]]

xgb_normalized += xgb_normalized[:1]
kmeans_gbdt_normalized += kmeans_gbdt_normalized[:1]

# 绘制
ax4.plot(angles, xgb_normalized, 'o-', linewidth=2, label='XGBoost', color='#FF6B6B')
ax4.fill(angles, xgb_normalized, alpha=0.25, color='#FF6B6B')
ax4.plot(angles, kmeans_gbdt_normalized, 'o-', linewidth=2, label='KMeans-GBDT', color='#4ECDC4')
ax4.fill(angles, kmeans_gbdt_normalized, alpha=0.25, color='#4ECDC4')

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, fontsize=11)
ax4.set_ylim(0, 1)
ax4.set_title('模型性能雷达图对比（归一化）', fontsize=14, weight='bold', pad=20)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax4.grid(True)

plt.tight_layout()
plt.savefig('model_comparison_radar.png', dpi=300, bbox_inches='tight')
print("  已保存: model_comparison_radar.png")
plt.close()

# 图5：改进幅度对比（显示KMeans-GBDT相对于XGBoost的提升百分比）
fig5, ax5 = plt.subplots(figsize=(10, 6))
improvements = [
    (f1_comparison_20m[1] - f1_comparison_20m[0]) / f1_comparison_20m[0] * 100 if f1_comparison_20m[0] > 0 else 0,
    (precision_comparison_20m[1] - precision_comparison_20m[0]) / precision_comparison_20m[0] * 100 if precision_comparison_20m[0] > 0 else 0,
    (recall_comparison_20m[1] - recall_comparison_20m[0]) / recall_comparison_20m[0] * 100 if recall_comparison_20m[0] > 0 else 0
]

bars = ax5.bar(categories, improvements, color=['#4472C4', '#70AD47', '#FFC000'], alpha=0.8)
ax5.set_ylabel('提升百分比 (%)', fontsize=12)
ax5.set_title('KMeans-GBDT相对于XGBoost的性能提升', fontsize=14, weight='bold')
ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax5.grid(axis='y', alpha=0.3, linestyle='--')

# 添加数值标签
for bar, imp in zip(bars, improvements):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{imp:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig('model_improvement.png', dpi=300, bbox_inches='tight')
print("  已保存: model_improvement.png")
plt.close()

print("\n所有图表生成完成！")

