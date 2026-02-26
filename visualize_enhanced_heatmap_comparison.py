#!/usr/bin/env python3
"""
Create heatmaps showing model performance across validation scenarios
Visualizes how different validation methods affect accuracy and dataset size
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Model configuration
MODEL_INFO = {
    'gemini-2-5-pro': {'label': 'Gemini 2.5 Pro', 'color': '#4285F4'},
    'gemini-2-5-flash': {'label': 'Gemini 2.5 Flash', 'color': '#34A853'},
    'claude-opus-4-1': {'label': 'Claude Opus 4', 'color': '#EA4335'},
    'claude-sonnet-4-5': {'label': 'Claude Sonnet 4.5', 'color': '#FBBC04'}
}


def create_absolute_accuracy_heatmap():
    """Heatmap showing absolute conditional accuracy across all scenarios"""

    df = pd.read_csv('enhanced_fair_comparison_with_judge.csv')

    # All scenarios
    scenarios = [
        'Baseline',
        'Tier 1 Only', 'Tier 1 ∩ Judge', 'Tier 1 ∪ Judge',
        'Tier 1-2 Only', 'Tier 1-2 ∩ Judge', 'Tier 1-2 ∪ Judge',
        'Tier 1-3 Only', 'Tier 1-3 ∩ Judge', 'Tier 1-3 ∪ Judge',
        'Tier 1-4 Only', 'Tier 1-4 ∩ Judge', 'Tier 1-4 ∪ Judge',
        'Judge Only'
    ]

    scenario_labels = [
        'Base',
        'T1\nOnly', 'T1\n∩J', 'T1\n∪J',
        'T1-2\nOnly', 'T1-2\n∩J', 'T1-2\n∪J',
        'T1-3\nOnly', 'T1-3\n∩J', 'T1-3\n∪J',
        'T1-4\nOnly', 'T1-4\n∩J', 'T1-4\n∪J',
        'Judge\nOnly'
    ]

    models = list(MODEL_INFO.keys())
    model_labels = [MODEL_INFO[m]['label'] for m in models]

    # Build matrix
    acc_matrix = []
    for model_id in models:
        model_accs = []
        for scenario in scenarios:
            data = df[(df['scenario'] == scenario) & (df['model'] == model_id)]
            if not data.empty:
                model_accs.append(data['conditional_accuracy'].values[0])
            else:
                model_accs.append(0)
        acc_matrix.append(model_accs)

    acc_matrix = np.array(acc_matrix)

    fig, ax = plt.subplots(figsize=(16, 6))

    # Create heatmap
    im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=50, vmax=95)

    # Set ticks
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(scenario_labels, fontsize=9)
    ax.set_yticklabels(model_labels, fontsize=12, fontweight='bold')

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(scenarios)):
            value = acc_matrix[i, j]
            text_color = 'white' if value < 72 else 'black'
            text = ax.text(j, i, f'{value:.1f}',
                          ha="center", va="center", color=text_color,
                          fontsize=8, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Conditional Accuracy (%)', fontsize=12, fontweight='bold')

    ax.set_title('Absolute Conditional Accuracy Across All Validation Scenarios\n'
                'Union (∪) Achieves Highest Accuracy for Each Tier Level',
                fontsize=14, fontweight='bold', pad=20)

    # Add vertical lines to separate tier groups
    for pos in [3.5, 6.5, 9.5, 12.5]:
        ax.axvline(x=pos, color='white', linestyle='-', linewidth=3)

    # Add tier labels
    tier_positions = [2, 5, 8, 11]
    tier_labels = ['Tier 1', 'Tier 1-2', 'Tier 1-3', 'Tier 1-4']
    for pos, label in zip(tier_positions, tier_labels):
        ax.text(pos, -0.7, label, ha='center', va='top',
               fontsize=10, fontweight='bold', style='italic',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    output_file = 'enhanced_heatmap_absolute_accuracy.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved absolute accuracy heatmap: {output_file}")
    plt.close()


def create_gain_from_baseline_heatmap():
    """Heatmap showing accuracy gains from baseline"""

    df = pd.read_csv('enhanced_fair_comparison_with_judge.csv')

    scenarios = [
        'Tier 1 Only', 'Tier 1 ∩ Judge', 'Tier 1 ∪ Judge',
        'Tier 1-2 Only', 'Tier 1-2 ∩ Judge', 'Tier 1-2 ∪ Judge',
        'Tier 1-3 Only', 'Tier 1-3 ∩ Judge', 'Tier 1-3 ∪ Judge',
        'Tier 1-4 Only', 'Tier 1-4 ∩ Judge', 'Tier 1-4 ∪ Judge',
        'Judge Only'
    ]

    scenario_labels = [
        'T1\nOnly', 'T1\n∩J', 'T1\n∪J',
        'T1-2\nOnly', 'T1-2\n∩J', 'T1-2\n∪J',
        'T1-3\nOnly', 'T1-3\n∩J', 'T1-3\n∪J',
        'T1-4\nOnly', 'T1-4\n∩J', 'T1-4\n∪J',
        'Judge\nOnly'
    ]

    models = list(MODEL_INFO.keys())
    model_labels = [MODEL_INFO[m]['label'] for m in models]

    # Build gains matrix
    gains_matrix = []
    for model_id in models:
        baseline_acc = df[(df['scenario'] == 'Baseline') &
                         (df['model'] == model_id)]['conditional_accuracy'].values[0]

        model_gains = []
        for scenario in scenarios:
            data = df[(df['scenario'] == scenario) & (df['model'] == model_id)]
            if not data.empty:
                scenario_acc = data['conditional_accuracy'].values[0]
                gain = scenario_acc - baseline_acc
                model_gains.append(gain)
            else:
                model_gains.append(0)

        gains_matrix.append(model_gains)

    gains_matrix = np.array(gains_matrix)

    fig, ax = plt.subplots(figsize=(15, 6))

    # Create heatmap
    vmax = max(gains_matrix.max(), 25)
    im = ax.imshow(gains_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=0, vmax=vmax)

    # Set ticks
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(scenario_labels, fontsize=9)
    ax.set_yticklabels(model_labels, fontsize=12, fontweight='bold')

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(scenarios)):
            value = gains_matrix[i, j]
            text_color = 'white' if value < vmax * 0.5 else 'black'
            text = ax.text(j, i, f'+{value:.1f}',
                          ha="center", va="center", color=text_color,
                          fontsize=8, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy Gain from Baseline (pp)', fontsize=12, fontweight='bold')

    ax.set_title('Accuracy Gains from Baseline Across Validation Scenarios\n'
                'Higher Gains (Green) Indicate Better Question Quality Filtering',
                fontsize=14, fontweight='bold', pad=20)

    # Add vertical lines
    for pos in [2.5, 5.5, 8.5, 11.5]:
        ax.axvline(x=pos, color='white', linestyle='-', linewidth=3)

    # Add tier labels
    tier_positions = [1, 4, 7, 10]
    tier_labels = ['Tier 1', 'Tier 1-2', 'Tier 1-3', 'Tier 1-4']
    for pos, label in zip(tier_positions, tier_labels):
        ax.text(pos, -0.7, label, ha='center', va='top',
               fontsize=10, fontweight='bold', style='italic',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    output_file = 'enhanced_heatmap_gains_from_baseline.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved gains from baseline heatmap: {output_file}")
    plt.close()


def create_quality_coverage_heatmap():
    """Heatmap showing dataset size vs accuracy tradeoff"""

    df = pd.read_csv('enhanced_fair_comparison_with_judge.csv')

    # Select representative scenarios
    scenarios = [
        'Baseline',
        'Tier 1 Only', 'Tier 1 ∪ Judge',
        'Tier 1-2 Only', 'Tier 1-2 ∪ Judge',
        'Tier 1-3 Only', 'Tier 1-3 ∪ Judge',
        'Tier 1-4 Only', 'Tier 1-4 ∪ Judge',
        'Judge Only'
    ]

    scenario_labels = [
        'Baseline\n(670)',
        'T1 Only\n(637)', 'T1 ∪ J\n(531)',
        'T1-2 Only\n(563)', 'T1-2 ∪ J\n(493)',
        'T1-3 Only\n(513)', 'T1-3 ∪ J\n(457)',
        'T1-4 Only\n(419)', 'T1-4 ∪ J\n(386)',
        'Judge Only\n(540)'
    ]

    models = list(MODEL_INFO.keys())
    model_labels = [MODEL_INFO[m]['label'] for m in models]

    # Build matrix
    acc_matrix = []
    size_matrix = []

    for model_id in models:
        model_accs = []
        model_sizes = []
        for scenario in scenarios:
            data = df[(df['scenario'] == scenario) & (df['model'] == model_id)]
            if not data.empty:
                model_accs.append(data['conditional_accuracy'].values[0])
                model_sizes.append(data['evaluated_count'].values[0])
            else:
                model_accs.append(0)
                model_sizes.append(0)
        acc_matrix.append(model_accs)
        size_matrix.append(model_sizes)

    acc_matrix = np.array(acc_matrix)
    size_matrix = np.array(size_matrix)

    # Create dual heatmap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Top: Accuracy
    im1 = ax1.imshow(acc_matrix, cmap='RdYlGn', aspect='auto',
                     vmin=50, vmax=95)

    ax1.set_xticks(np.arange(len(scenarios)))
    ax1.set_yticks(np.arange(len(models)))
    ax1.set_xticklabels(scenario_labels, fontsize=9)
    ax1.set_yticklabels(model_labels, fontsize=11, fontweight='bold')

    for i in range(len(models)):
        for j in range(len(scenarios)):
            value = acc_matrix[i, j]
            text_color = 'white' if value < 72 else 'black'
            ax1.text(j, i, f'{value:.1f}%',
                    ha="center", va="center", color=text_color,
                    fontsize=7, fontweight='bold')

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Conditional Accuracy (%)', fontsize=10, fontweight='bold')

    ax1.set_title('Quality vs Coverage Tradeoff: Accuracy',
                 fontsize=13, fontweight='bold', pad=15)

    # Bottom: Dataset Size
    im2 = ax2.imshow(size_matrix, cmap='Blues', aspect='auto',
                     vmin=350, vmax=670)

    ax2.set_xticks(np.arange(len(scenarios)))
    ax2.set_yticks(np.arange(len(models)))
    ax2.set_xticklabels(scenario_labels, fontsize=9)
    ax2.set_yticklabels(model_labels, fontsize=11, fontweight='bold')

    for i in range(len(models)):
        for j in range(len(scenarios)):
            value = size_matrix[i, j]
            text_color = 'white' if value < 510 else 'black'
            ax2.text(j, i, f'{int(value)}',
                    ha="center", va="center", color=text_color,
                    fontsize=7, fontweight='bold')

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Questions Evaluated', fontsize=10, fontweight='bold')

    ax2.set_title('Quality vs Coverage Tradeoff: Dataset Size',
                 fontsize=13, fontweight='bold', pad=15)

    # Add vertical lines
    for ax in [ax1, ax2]:
        for pos in [2.5, 4.5, 6.5, 8.5]:
            ax.axvline(x=pos, color='white', linestyle='-', linewidth=2)

    plt.suptitle('Quality vs Coverage Tradeoff\n'
                 'Top: Accuracy (higher is better)  |  Bottom: Dataset Size (higher = more data)',
                 fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout()
    output_file = 'enhanced_heatmap_quality_coverage_tradeoff.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved quality-coverage tradeoff heatmap: {output_file}")
    plt.close()


def create_recommended_scenarios_heatmap():
    """Compact heatmap comparing only recommended scenarios"""

    df = pd.read_csv('enhanced_fair_comparison_with_judge.csv')

    scenarios = [
        'Baseline',
        'Judge Only',
        'Tier 1-2 ∪ Judge',
        'Tier 1-4 ∪ Judge'
    ]

    scenario_labels = [
        'Baseline\n(670 Q)\n68.0%*',
        'Judge Only\n(540 Q)\n81.7%*',
        'Tier 1-2 ∪ Judge\n(493 Q)\n87.1%*\n(RECOMMENDED)',
        'Tier 1-4 ∪ Judge\n(386 Q)\n93.6%*\n(MAX QUALITY)'
    ]

    models = list(MODEL_INFO.keys())
    model_labels = [MODEL_INFO[m]['label'] for m in models]

    # Build matrices for three metrics
    metrics = [
        ('response_rate', 'Response Rate (%)', 'Blues', 94, 99),
        ('accuracy', 'Overall Accuracy (%)', 'Greens', 45, 95),
        ('conditional_accuracy', 'Conditional Accuracy (%)', 'RdYlGn', 50, 95)
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    for idx, (metric_key, metric_label, cmap, vmin, vmax) in enumerate(metrics):
        ax = axes[idx]

        # Build matrix
        metric_matrix = []
        for model_id in models:
            model_values = []
            for scenario in scenarios:
                data = df[(df['scenario'] == scenario) & (df['model'] == model_id)]
                if not data.empty:
                    model_values.append(data[metric_key].values[0])
                else:
                    model_values.append(0)
            metric_matrix.append(model_values)

        metric_matrix = np.array(metric_matrix)

        # Create heatmap
        im = ax.imshow(metric_matrix, cmap=cmap, aspect='auto',
                      vmin=vmin, vmax=vmax)

        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(scenario_labels, fontsize=10, fontweight='bold')
        ax.set_yticklabels(model_labels, fontsize=11, fontweight='bold')

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(scenarios)):
                value = metric_matrix[i, j]
                # Dynamic text color
                if cmap == 'Blues':
                    text_color = 'white' if value < 96.5 else 'black'
                elif cmap == 'Greens':
                    text_color = 'white' if value < 70 else 'black'
                else:  # RdYlGn
                    text_color = 'white' if value < 72 else 'black'

                ax.text(j, i, f'{value:.1f}',
                       ha="center", va="center", color=text_color,
                       fontsize=9, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(metric_label, fontsize=10, fontweight='bold')

        ax.set_title(metric_label, fontsize=12, fontweight='bold', pad=10)

        # Add grid
        ax.set_xticks(np.arange(len(scenarios)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(models)) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=2)

    plt.suptitle('Recommended Validation Scenarios Comparison\n'
                 '* Gemini 2.5 Pro conditional accuracy shown in labels',
                 fontsize=15, fontweight='bold', y=0.995)

    plt.tight_layout()
    output_file = 'enhanced_heatmap_recommended_scenarios.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved recommended scenarios heatmap: {output_file}")
    plt.close()


def main():
    print("="*80)
    print("Generating Enhanced Heatmap Comparison Charts")
    print("="*80)
    print()

    print("Creating absolute accuracy heatmap...")
    create_absolute_accuracy_heatmap()

    print("Creating gains from baseline heatmap...")
    create_gain_from_baseline_heatmap()

    print("Creating quality-coverage tradeoff heatmap...")
    create_quality_coverage_heatmap()

    print("Creating recommended scenarios heatmap...")
    create_recommended_scenarios_heatmap()

    print()
    print("="*80)
    print("✓ All enhanced heatmap charts generated successfully!")
    print("="*80)
    print()
    print("Generated files:")
    print("  - enhanced_heatmap_absolute_accuracy.png")
    print("  - enhanced_heatmap_gains_from_baseline.png")
    print("  - enhanced_heatmap_quality_coverage_tradeoff.png")
    print("  - enhanced_heatmap_recommended_scenarios.png")
    print()


if __name__ == "__main__":
    main()
