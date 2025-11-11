import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
from itertools import combinations

warnings.filterwarnings('ignore')

# Set publication-quality plotting parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.2)

print("="*80)
print("PAPER-QUALITY ANALYSIS - GATING MECHANISMS FOR EEG CHANNEL SELECTION")
print("="*80)
print()

# Configuration
RESULTS_DIR = Path('../kaggle_notebooks/results/physionet-gating-channel-selection-COMPLETE')
OUTPUT_DIR = Path('./paper_results')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Method definitions
METHODS = ['baseline', 'static', 'adaptive', 'halting']
METHOD_COLORS = {
    'baseline': '#2E86AB',
    'static': '#A23B72',
    'adaptive': '#06A77D',
    'halting': '#F18F01'
}
METHOD_LABELS = {
    'baseline': 'Baseline EEG-ARNN',
    'static': 'Static Gating',
    'adaptive': 'Adaptive Gating',
    'halting': 'Early Halting'
}

SELECTION_LABELS = {
    'ES': 'Edge Selection',
    'AS': 'Aggregation Selection',
    'GS': 'Gate Selection'
}

# Load all results
print("Loading experimental results...")
results = {}
retrain_results = {}

for method in METHODS:
    results_file = RESULTS_DIR / f'{method}_results.csv'
    retrain_file = RESULTS_DIR / f'{method}_retrain_results.csv'

    if results_file.exists():
        results[method] = pd.read_csv(results_file)
        print(f"  Loaded {METHOD_LABELS[method]}: {len(results[method])} subjects")

    if retrain_file.exists():
        retrain_results[method] = pd.read_csv(retrain_file)

subjects = sorted(results['baseline']['subject'].unique())
print(f"\nTotal subjects: {len(subjects)}")

# ============================================================================
# TABLE 1: Overall Performance Summary
# ============================================================================
print("\n" + "="*80)
print("GENERATING TABLE 1: Overall Performance Summary")
print("="*80)

summary_stats = []
for method in METHODS:
    if method in results:
        df = results[method]
        summary_stats.append({
            'Method': METHOD_LABELS[method],
            'Mean (%)': df['accuracy'].mean() * 100,
            'Std (%)': df['accuracy'].std() * 100,
            'Min (%)': df['accuracy'].min() * 100,
            'Max (%)': df['accuracy'].max() * 100,
            'Median (%)': df['accuracy'].median() * 100,
            'IQR (%)': (df['accuracy'].quantile(0.75) - df['accuracy'].quantile(0.25)) * 100,
            'N': len(df)
        })

summary_df = pd.DataFrame(summary_stats)
summary_df = summary_df.sort_values('Mean (%)', ascending=False)

print("\n" + summary_df.to_string(index=False, float_format='%.2f'))

summary_df.to_csv(OUTPUT_DIR / 'table1_overall_performance.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'table1_overall_performance.csv'}")

# ============================================================================
# TABLE 2: Statistical Significance Testing
# ============================================================================
print("\n" + "="*80)
print("GENERATING TABLE 2: Statistical Significance Tests")
print("="*80)

pairwise_results = []
for method1, method2 in combinations(METHODS, 2):
    if method1 in results and method2 in results:
        acc1 = results[method1].sort_values('subject')['accuracy'].values
        acc2 = results[method2].sort_values('subject')['accuracy'].values

        t_stat, p_value = stats.ttest_rel(acc1, acc2)
        diff = acc1 - acc2
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)

        pairwise_results.append({
            'Comparison': f"{METHOD_LABELS[method1]} vs {METHOD_LABELS[method2]}",
            'Mean Diff (%)': np.mean(diff) * 100,
            't-statistic': t_stat,
            'p-value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No',
            "Cohen's d": cohens_d
        })

pairwise_df = pd.DataFrame(pairwise_results)
print("\n" + pairwise_df.to_string(index=False, float_format='%.4f'))

pairwise_df.to_csv(OUTPUT_DIR / 'table2_statistical_tests.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'table2_statistical_tests.csv'}")

# Friedman test
print("\nFriedman Test:")
accuracy_matrix = np.column_stack([results[m].sort_values('subject')['accuracy'].values for m in METHODS])
friedman_stat, friedman_p = stats.friedmanchisquare(*accuracy_matrix.T)
print(f"  Chi-square = {friedman_stat:.4f}, p-value = {friedman_p:.4f}")
print(f"  {'Significant' if friedman_p < 0.05 else 'Not significant'} differences between methods")

# ============================================================================
# TABLE 3: Subject-Wise Comparison
# ============================================================================
print("\n" + "="*80)
print("GENERATING TABLE 3: Subject-Wise Performance")
print("="*80)

subject_comparison = []
for subject in subjects:
    row = {'Subject': subject}
    for method in METHODS:
        if method in results:
            subj_data = results[method][results[method]['subject'] == subject]
            if len(subj_data) > 0:
                row[METHOD_LABELS[method]] = subj_data['accuracy'].values[0] * 100

    accuracies = [(METHOD_LABELS[m], row.get(METHOD_LABELS[m], 0)) for m in METHODS]
    best_method, best_acc = max(accuracies, key=lambda x: x[1])
    row['Best Method'] = best_method
    row['Range (%)'] = max([row.get(METHOD_LABELS[m], 0) for m in METHODS]) - min([row.get(METHOD_LABELS[m], 0) for m in METHODS])

    subject_comparison.append(row)

subject_df = pd.DataFrame(subject_comparison)
print("\n" + subject_df.to_string(index=False, float_format='%.2f'))

method_wins = subject_df['Best Method'].value_counts()
print("\nMethod Wins:")
for method, count in method_wins.items():
    print(f"  {method}: {count} subjects ({count/len(subjects)*100:.1f}%)")

subject_df.to_csv(OUTPUT_DIR / 'table3_subject_wise.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'table3_subject_wise.csv'}")

# ============================================================================
# TABLE 4: Channel Selection Performance
# ============================================================================
print("\n" + "="*80)
print("GENERATING TABLE 4: Channel Selection Performance")
print("="*80)

channel_sel_summary = []
for method in METHODS:
    if method not in retrain_results:
        continue

    df = retrain_results[method]
    selection_methods = df['method'].unique()

    for sel_method in selection_methods:
        sel_df = df[df['method'] == sel_method]
        channel_sel_summary.append({
            'Gating Method': METHOD_LABELS[method],
            'Selection': SELECTION_LABELS.get(sel_method, sel_method),
            'Mean Drop (%)': sel_df['accuracy_drop_pct'].mean(),
            'Std Drop (%)': sel_df['accuracy_drop_pct'].std(),
            'Min Drop (%)': sel_df['accuracy_drop_pct'].min(),
            'Max Drop (%)': sel_df['accuracy_drop_pct'].max(),
            'N': len(sel_df)
        })

channel_sel_df = pd.DataFrame(channel_sel_summary)
channel_sel_df = channel_sel_df.sort_values('Mean Drop (%)')

print("\n" + channel_sel_df.to_string(index=False, float_format='%.2f'))

channel_sel_df.to_csv(OUTPUT_DIR / 'table4_channel_selection.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'table4_channel_selection.csv'}")

# ============================================================================
# TABLE 5: Performance by K
# ============================================================================
print("\n" + "="*80)
print("GENERATING TABLE 5: Performance vs Number of Channels")
print("="*80)

k_values = [10, 15, 20, 25, 30]
k_performance = []

for k in k_values:
    row = {'K': k, 'Reduction (%)': (64-k)/64*100}

    for method in METHODS:
        if method not in retrain_results:
            continue

        df = retrain_results[method]
        k_df = df[df['k'] == k]

        if len(k_df) > 0:
            row[f'{METHOD_LABELS[method]} Acc (%)'] = k_df['accuracy'].mean() * 100
            row[f'{METHOD_LABELS[method]} Drop (%)'] = k_df['accuracy_drop_pct'].mean()

    k_performance.append(row)

k_perf_df = pd.DataFrame(k_performance)
print("\n" + k_perf_df.to_string(index=False, float_format='%.2f'))

k_perf_df.to_csv(OUTPUT_DIR / 'table5_k_performance.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'table5_k_performance.csv'}")

# ============================================================================
# FIGURE 1: Overall Performance Comparison
# ============================================================================
print("\n" + "="*80)
print("GENERATING FIGURE 1: Overall Performance Comparison")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (a) Bar plot
ax = axes[0]
means = [results[m]['accuracy'].mean() * 100 for m in METHODS if m in results]
stds = [results[m]['accuracy'].std() * 100 for m in METHODS if m in results]
labels = [METHOD_LABELS[m] for m in METHODS if m in results]
colors_list = [METHOD_COLORS[m] for m in METHODS if m in results]

x_pos = np.arange(len(labels))
bars = ax.bar(x_pos, means, yerr=stds, color=colors_list, alpha=0.8,
              capsize=5, edgecolor='black', linewidth=1.2)

ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_xlabel('Method', fontweight='bold')
ax.set_title('(a) Mean Classification Accuracy', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=15, ha='right')
ax.set_ylim([70, 90])
ax.grid(axis='y', alpha=0.3, linestyle='--')

for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, mean + std + 0.8, f'{mean:.1f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')

# (b) Box plot
ax = axes[1]
data_for_box = [results[m]['accuracy'].values * 100 for m in METHODS if m in results]
bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True,
                showmeans=True, meanline=True,
                meanprops=dict(color='red', linewidth=2),
                medianprops=dict(color='black', linewidth=2))

for patch, method in zip(bp['boxes'], [m for m in METHODS if m in results]):
    patch.set_facecolor(METHOD_COLORS[method])
    patch.set_alpha(0.8)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.2)

ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_xlabel('Method', fontweight='bold')
ax.set_title('(b) Distribution Across Subjects', fontweight='bold')
ax.set_xticklabels(labels, rotation=15, ha='right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([70, 100])

# (c) Wins
ax = axes[2]
win_counts = [method_wins.get(METHOD_LABELS[m], 0) for m in METHODS]
colors_list = [METHOD_COLORS[m] for m in METHODS]

wedges, texts, autotexts = ax.pie(win_counts, labels=labels,
                                   autopct='%1.0f%%', colors=colors_list,
                                   startangle=90, textprops={'fontsize': 9})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax.set_title('(c) Best Method Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure1_overall_performance.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'figure1_overall_performance.pdf', bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'figure1_overall_performance.png'}")
print(f"Saved: {OUTPUT_DIR / 'figure1_overall_performance.pdf'}")
plt.close()

# ============================================================================
# FIGURE 2: Subject Heatmap
# ============================================================================
print("\n" + "="*80)
print("GENERATING FIGURE 2: Subject-Wise Heatmap")
print("="*80)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

subject_matrix = []
for subject in subjects:
    row = []
    for method in METHODS:
        if method in results:
            subj_data = results[method][results[method]['subject'] == subject]
            if len(subj_data) > 0:
                row.append(subj_data['accuracy'].values[0] * 100)
            else:
                row.append(np.nan)
    subject_matrix.append(row)

subject_matrix = np.array(subject_matrix)

im = ax.imshow(subject_matrix, aspect='auto', cmap='RdYlGn',
               vmin=70, vmax=95, interpolation='nearest')

for i in range(len(subjects)):
    for j in range(len(METHODS)):
        text = ax.text(j, i, f'{subject_matrix[i, j]:.1f}',
                      ha="center", va="center", color="black", fontsize=8)

ax.set_yticks(range(len(subjects)))
ax.set_yticklabels(subjects, fontsize=10)
ax.set_xticks(range(len(METHODS)))
ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=45, ha='right', fontsize=10)
ax.set_title('Subject-wise Classification Accuracy (%)', fontweight='bold', fontsize=12, pad=15)
ax.set_xlabel('Method', fontweight='bold', fontsize=11)
ax.set_ylabel('Subject', fontweight='bold', fontsize=11)

cbar = plt.colorbar(im, ax=ax, label='Accuracy (%)')
cbar.set_label('Accuracy (%)', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure2_subject_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'figure2_subject_heatmap.pdf', bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'figure2_subject_heatmap.png'}")
print(f"Saved: {OUTPUT_DIR / 'figure2_subject_heatmap.pdf'}")
plt.close()

# ============================================================================
# FIGURE 3: Channel Selection Performance
# ============================================================================
print("\n" + "="*80)
print("GENERATING FIGURE 3: Channel Selection Performance")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, method in enumerate(METHODS):
    if method not in retrain_results:
        continue

    ax = axes[idx]
    df = retrain_results[method]

    selection_methods = sorted(df['method'].unique())
    markers = {'ES': 'o', 'AS': 's', 'GS': '^'}
    colors = {'ES': '#E63946', 'AS': '#457B9D', 'GS': '#2A9D8F'}

    for sel_method in selection_methods:
        sel_df = df[df['method'] == sel_method]
        k_vals = sorted(sel_df['k'].unique())

        accuracies = []
        stds = []
        for k in k_vals:
            k_df = sel_df[sel_df['k'] == k]
            accuracies.append(k_df['accuracy'].mean() * 100)
            stds.append(k_df['accuracy'].std() * 100)

        ax.plot(k_vals, accuracies, marker=markers.get(sel_method, 'o'),
               label=SELECTION_LABELS.get(sel_method, sel_method),
               color=colors.get(sel_method, 'gray'),
               linewidth=2.5, markersize=8, alpha=0.8)

        ax.fill_between(k_vals,
                       np.array(accuracies) - np.array(stds),
                       np.array(accuracies) + np.array(stds),
                       color=colors.get(sel_method, 'gray'), alpha=0.15)

    full_acc = df['full_channels_acc'].iloc[0] * 100
    ax.axhline(y=full_acc, color='black', linestyle='--', linewidth=2,
              label=f'Full (64 ch): {full_acc:.1f}%', alpha=0.7)

    ax.set_xlabel('Number of Channels (k)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=11)
    ax.set_title(f'({chr(97+idx)}) {METHOD_LABELS[method]}', fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([70, 95])
    ax.set_xticks(k_vals)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure3_channel_selection.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'figure3_channel_selection.pdf', bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'figure3_channel_selection.png'}")
print(f"Saved: {OUTPUT_DIR / 'figure3_channel_selection.pdf'}")
plt.close()

# ============================================================================
# FIGURE 4: Accuracy Drop
# ============================================================================
print("\n" + "="*80)
print("GENERATING FIGURE 4: Accuracy Drop Comparison")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

selection_method_types = ['ES', 'AS', 'GS']
titles = ['(a) Edge Selection (ES)', '(b) Aggregation Selection (AS)', '(c) Gate Selection (GS)']

for idx, (sel_method, title) in enumerate(zip(selection_method_types, titles)):
    ax = axes[idx]

    for method in METHODS:
        if method not in retrain_results:
            continue

        df = retrain_results[method]
        sel_df = df[df['method'] == sel_method]

        if len(sel_df) == 0:
            continue

        k_values = sorted(sel_df['k'].unique())
        mean_drops = []
        std_drops = []

        for k in k_values:
            k_df = sel_df[sel_df['k'] == k]
            mean_drops.append(k_df['accuracy_drop_pct'].mean())
            std_drops.append(k_df['accuracy_drop_pct'].std())

        ax.plot(k_values, mean_drops, 'o-', label=METHOD_LABELS[method],
               color=METHOD_COLORS[method], linewidth=2.5, markersize=8, alpha=0.8)
        ax.fill_between(k_values,
                       np.array(mean_drops) - np.array(std_drops),
                       np.array(mean_drops) + np.array(std_drops),
                       color=METHOD_COLORS[method], alpha=0.15)

    ax.set_xlabel('Number of Channels (k)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Accuracy Drop (%)', fontweight='bold', fontsize=11)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.set_xticks(k_values)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure4_accuracy_drop.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'figure4_accuracy_drop.pdf', bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'figure4_accuracy_drop.png'}")
print(f"Saved: {OUTPUT_DIR / 'figure4_accuracy_drop.pdf'}")
plt.close()

# ============================================================================
# FIGURE 5: Retention Analysis
# ============================================================================
print("\n" + "="*80)
print("GENERATING FIGURE 5: Accuracy Retention Analysis")
print("="*80)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for method in METHODS:
    if method not in retrain_results:
        continue

    df = retrain_results[method]
    k_values_sorted = sorted(df['k'].unique())
    retention_rates = []
    retention_stds = []

    for k in k_values_sorted:
        k_df = df[df['k'] == k]
        retentions = (k_df['accuracy'] / k_df['full_channels_acc']) * 100
        retention_rates.append(retentions.mean())
        retention_stds.append(retentions.std())

    ax.plot(k_values_sorted, retention_rates, 'o-', label=METHOD_LABELS[method],
           color=METHOD_COLORS[method], linewidth=2.5, markersize=8, alpha=0.8)
    ax.fill_between(k_values_sorted,
                   np.array(retention_rates) - np.array(retention_stds),
                   np.array(retention_rates) + np.array(retention_stds),
                   color=METHOD_COLORS[method], alpha=0.15)

ax.axhline(y=100, color='black', linestyle='--', linewidth=2,
          label='Full Channel Performance', alpha=0.7)
ax.axhline(y=95, color='red', linestyle=':', linewidth=1.5,
          label='95% Retention Threshold', alpha=0.7)

ax.set_xlabel('Number of Channels (k)', fontweight='bold', fontsize=12)
ax.set_ylabel('Accuracy Retention (%)', fontweight='bold', fontsize=12)
ax.set_title('Accuracy Retention vs Channel Reduction', fontweight='bold', fontsize=13)
ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim([85, 105])
ax.set_xticks(k_values_sorted)

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks([10, 15, 20, 25, 30])
ax2.set_xticklabels([f'{(64-k)/64*100:.0f}%' for k in [10, 15, 20, 25, 30]])
ax2.set_xlabel('Channel Reduction', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure5_retention.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'figure5_retention.pdf', bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'figure5_retention.png'}")
print(f"Saved: {OUTPUT_DIR / 'figure5_retention.pdf'}")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print()

winner = summary_df.iloc[0]
best_sel = channel_sel_df.iloc[0]

print("KEY RESULTS:")
print(f"  Overall Winner: {winner['Method']}")
print(f"    Mean Accuracy: {winner['Mean (%)']:.2f}% ± {winner['Std (%)']:.2f}%")
print()
print(f"  Channel Selection Winner: {best_sel['Gating Method']} + {best_sel['Selection']}")
print(f"    Mean Drop: {best_sel['Mean Drop (%)']:.2f}%")
print()
print(f"All results saved to: {OUTPUT_DIR.absolute()}")
print()
print("Generated Files:")
print("  Tables (CSV): table1-5")
print("  Figures (PNG + PDF): figure1-5")
