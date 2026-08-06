"""
优化版评估脚本 (Evaluate Optimized)
====================================
加载预测结果并与三个基线后端（最低RO/S1/CNOT错误率）对比。
用法: python evaluate_optimized.py

依赖:
  - results_optimized/predictions_optimized.csv
  - results_optimized/trained_model_optimized.pkl (含test_df、baseline后端信息)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / 'results'

print("=" * 60)
print("Optimized Evaluation (evaluate_optimized.py)")
print("=" * 60)

# ============================================================================
# 1. 加载模型和预测结果
# ============================================================================
print("\n[1] 加载模型工件...")
with open(OUTPUT_DIR / 'trained_model_optimized.pkl', 'rb') as f:
    model_artifacts = pickle.load(f)

lowest_ro_backend = model_artifacts['lowest_ro_backend']
lowest_s1_backend = model_artifacts['lowest_s1_backend']
lowest_cnot_backend = model_artifacts['lowest_cnot_backend']
test_df = model_artifacts['test_df']

print(f"  基线后端:")
print(f"    最低RO错误:   {lowest_ro_backend}")
print(f"    最低S1错误:   {lowest_s1_backend}")
print(f"    最低CNOT错误: {lowest_cnot_backend}")

print("\n[2] 加载预测结果...")
pred_df = pd.read_csv(OUTPUT_DIR / 'predictions_optimized.csv')
print(f"  预测数: {len(pred_df)}")

# ============================================================================
# 2. 解析预测结果
# ============================================================================
def parse_top3(s):
    s = s.strip('[]')
    items = s.split(',')
    return [x.strip().strip("'") for x in items if x.strip()]

results = []
for _, row in pred_df.iterrows():
    results.append({
        'circuit': row['circuit'],
        'true_best': row['true_best_backend'],
        'true_fom': row['true_fom'],
        'pred_best': row['pred_best_backend'],
        'pred_fom': row['pred_fom'],
        'actual_fom_selected': row.get('actual_fom_selected', row['true_fom']),
        'top3_backends': parse_top3(row['top3_backends']),
    })

# ============================================================================
# 3. 分析各基线
# ============================================================================
def analyze_baseline(baseline_backend, test_df, results):
    better = worse = equal = 0
    fom_diffs = []
    fom_diffs_pct = []
    worse_cases = []

    for r in results:
        circuit = r['circuit']
        pred_best = r['pred_best']

        circuit_data = test_df[test_df['Circuit'] == circuit]
        baseline_row = circuit_data[circuit_data['Backend'] == baseline_backend]
        if len(baseline_row) == 0:
            continue
        baseline_fom = baseline_row['FOM'].iloc[0]

        pred_row = circuit_data[circuit_data['Backend'] == pred_best]
        if len(pred_row) == 0:
            continue
        pred_fom = pred_row['FOM'].iloc[0]

        true_best_fom = r['true_fom']

        if pred_fom > baseline_fom + 1e-6:
            better += 1
        elif pred_fom < baseline_fom - 1e-6:
            worse += 1
            diff = pred_fom - baseline_fom
            diff_pct = (diff / baseline_fom * 100) if baseline_fom > 0 else 0
            worse_cases.append({
                'circuit': circuit,
                'baseline': baseline_backend,
                'baseline_fom': baseline_fom,
                'pred_backend': pred_best,
                'pred_fom': pred_fom,
                'true_best': r['true_best'],
                'true_best_fom': true_best_fom,
                'fom_diff': diff,
                'fom_diff_pct': diff_pct,
            })
        else:
            equal += 1

        fom_diffs.append(pred_fom - baseline_fom)
        if baseline_fom > 0:
            fom_diffs_pct.append(((pred_fom - baseline_fom) / baseline_fom) * 100)
        else:
            fom_diffs_pct.append(0.0)

    total = len(fom_diffs)
    return {
        'baseline': baseline_backend,
        'total': total,
        'better': better,
        'worse': worse,
        'equal': equal,
        'better_pct': better / total * 100 if total > 0 else 0,
        'worse_pct': worse / total * 100 if total > 0 else 0,
        'avg_fom_diff': np.mean(fom_diffs) if fom_diffs else 0,
        'avg_improvement_pct': np.mean(fom_diffs_pct) if fom_diffs_pct else 0,
        'worse_cases': worse_cases,
    }


result1 = analyze_baseline(lowest_ro_backend, test_df, results)
result2 = analyze_baseline(lowest_s1_backend, test_df, results)
result3 = analyze_baseline(lowest_cnot_backend, test_df, results)

# ============================================================================
# 4. 输出结果
# ============================================================================
print("\n" + "=" * 60)
print("Baseline Comparison Results")
print("=" * 60)
print(f"\n{'Baseline':<25} {'Better':<10} {'Worse':<10} {'Equal':<10} {'Avg FOM Diff':<15} {'Avg Improvement'}")
print("-" * 85)
for r in [result1, result2, result3]:
    print(f"{r['baseline']:<25} {r['better_pct']:>6.2f}%   {r['worse_pct']:>6.2f}%   {r['equal']:>6}     {r['avg_fom_diff']:>10.4f}     {r['avg_improvement_pct']:>8.2f}%")

avg_better = (result1['better_pct'] + result2['better_pct'] + result3['better_pct']) / 3
avg_diff = (result1['avg_fom_diff'] + result2['avg_fom_diff'] + result3['avg_fom_diff']) / 3
avg_impr = (result1['avg_improvement_pct'] + result2['avg_improvement_pct'] + result3['avg_improvement_pct']) / 3
print("-" * 85)
print(f"{'AVERAGE':<25} {avg_better:>6.2f}%   {'-':>6}     {'-':>6}     {avg_diff:>10.4f}     {avg_impr:>8.2f}%")

# ============================================================================
# 5. 汇总表
# ============================================================================
summary_df = pd.DataFrame([
    {'Baseline': r['baseline'], 'Better(%)': f"{r['better_pct']:.1f}",
     'Worse(%)': f"{r['worse_pct']:.1f}", 'Equal': r['equal'],
     'Avg FOM Diff': f"{r['avg_fom_diff']:.4f}",
     'Avg Improvement(%)': f"{r['avg_improvement_pct']:.2f}"}
    for r in [result1, result2, result3]
])
summary_df.to_csv(OUTPUT_DIR / 'baseline_comparison_optimized.csv', index=False, encoding='utf-8-sig')

# ============================================================================
# 6. 可视化
# ============================================================================
print("\n[3] 生成图表...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

baselines = [r['baseline'][:15] for r in [result1, result2, result3]]
better_pcts = [r['better_pct'] for r in [result1, result2, result3]]
improvements = [r['avg_improvement_pct'] for r in [result1, result2, result3]]

colors_bar = ['#3498db', '#2ecc71', '#e74c3c']

bars = axes[0].bar(baselines, better_pcts, color=colors_bar, edgecolor='white', linewidth=1.2)
axes[0].set_ylabel('Better than Baseline (%)')
axes[0].set_title('Prediction Performance vs Baselines')
axes[0].set_ylim(0, 100)
for bar, pct in zip(bars, better_pcts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

bars2 = axes[1].bar(baselines, improvements, color=colors_bar, edgecolor='white', linewidth=1.2)
axes[1].set_ylabel('Avg FOM Improvement (%)')
axes[1].set_title('Avg FOM Improvement vs Baselines')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
for bar, imp in zip(bars2, improvements):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{imp:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'baseline_comparison_optimized.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  图表: {OUTPUT_DIR / 'baseline_comparison_optimized.png'}")

# ============================================================================
# 7. 完整报告
# ============================================================================
with open(OUTPUT_DIR / 'baseline_analysis_optimized.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("Optimized Model Baseline Analysis Report\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"EXPERIMENT OVERVIEW:\n")
    f.write(f"  Model: Random Forest (threshold=0.005, weight=0.15)\n")
    f.write(f"  Features: 23 circuit + 20 backend + 13 interaction\n")
    f.write(f"  Total test circuits: {len(results)}\n\n")
    f.write(f"THREE BASELINE BACKENDS:\n")
    f.write(f"  1. Lowest Readout Error: {lowest_ro_backend}\n")
    f.write(f"  2. Lowest S1 Gate Error: {lowest_s1_backend}\n")
    f.write(f"  3. Lowest CNOT Error: {lowest_cnot_backend}\n\n")

    f.write("=" * 80 + "\n")
    f.write("DETAILED COMPARISON TABLE\n")
    f.write("=" * 80 + "\n")
    for r in [result1, result2, result3]:
        f.write(f"\n  Baseline: {r['baseline']}\n")
        f.write(f"    Total circuits: {r['total']}\n")
        f.write(f"    Better: {r['better']} ({r['better_pct']:.1f}%)\n")
        f.write(f"    Worse: {r['worse']} ({r['worse_pct']:.1f}%)\n")
        f.write(f"    Equal: {r['equal']}\n")
        f.write(f"    Avg FOM Diff: {r['avg_fom_diff']:.4f}\n")
        f.write(f"    Avg Improvement: {r['avg_improvement_pct']:.2f}%\n")

    # Worse cases
    all_worse = result1['worse_cases'] + result2['worse_cases'] + result3['worse_cases']
    if all_worse:
        f.write(f"\nWORSE CASES ANALYSIS:\n")
        f.write(f"  Total worse cases: {len(all_worse)}\n")
        f.write(f"  Unique circuits: {len(set(c['circuit'] for c in all_worse))}\n\n")
        for i, case in enumerate(all_worse[:20], 1):
            f.write(f"  {i}. Circuit: {case['circuit'][:50]}\n")
            f.write(f"     Baseline ({case['baseline']}): FOM={case['baseline_fom']:.4f}\n")
            f.write(f"     Predicted ({case['pred_backend']}): FOM={case['pred_fom']:.4f}\n")
            f.write(f"     True Best ({case['true_best']}): FOM={case['true_best_fom']:.4f}\n")
            f.write(f"     Loss: {case['fom_diff']:.4f}\n\n")

print(f"\n  报告: {OUTPUT_DIR / 'baseline_analysis_optimized.txt'}")
print(f"  汇总: {OUTPUT_DIR / 'baseline_comparison_optimized.csv'}")
print("\n" + "=" * 60)
print("Evaluation completed!")
print("=" * 60)