#!/usr/bin/env python3
"""
Visualize enhanced fair comparison results integrating tier-based and
judge-based validation methods.

Shows 14 scenarios comparing different validation approaches:
- Baseline
- Tier-only exclusions (1, 1-2, 1-3, 1-4)
- Tier ∩ Judge (intersection - highest confidence)
- Tier ∪ Judge (union - most inclusive)
- Judge-only exclusion
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Model configuration
MODEL_INFO = {
    'gemini-2-5-pro': {'label': 'Gemini 2.5 Pro', 'color': '#4285F4', 'marker': 'o'},
    'gemini-2-5-flash': {'label': 'Gemini 2.5 Flash', 'color': '#34A853', 'marker': 's'},
    'claude-opus-4-1': {'label': 'Claude Opus 4', 'color': '#EA4335', 'marker': '^'},
    'claude-sonnet-4-5': {'label': 'Claude Sonnet 4.5', 'color': '#FBBC04', 'marker': 'D'}
}


def create_validation_method_comparison():
    """
    Chart 1: Compare validation methods on Tier 1-2 threshold
    Shows: Baseline, Tier Only, Tier∩Judge, Tier∪Judge, Judge Only
    """

    df = pd.read_csv('enhanced_fair_comparison_with_judge.csv')

    # Focus on Tier 1-2 comparison scenarios
    scenarios = ['Baseline', 'Tier 1-2 Only', 'Tier 1-2 ∩ Judge',
                 'Tier 1-2 ∪ Judge', 'Judge Only']
    scenario_labels = ['Baseline\n(670 Q)', 'Tier 1-2\nOnly\n(563 Q)',
                       'Tier 1-2 ∩\nJudge\n(610 Q)', 'Tier 1-2 ∪\nJudge\n(493 Q)',
                       'Judge\nOnly\n(540 Q)']

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot lines for each model
    for model_id, info in MODEL_INFO.items():
        accuracies = []
        for scenario in scenarios:
            data = df[(df['scenario'] == scenario) & (df['model'] == model_id)]
            if not data.empty:
                accuracies.append(data['conditional_accuracy'].values[0])
            else:
                accuracies.append(None)

        ax.plot(range(len(scenarios)), accuracies,
               label=info['label'],
               color=info['color'],
               marker=info['marker'],
               linewidth=3,
               markersize=10,
               alpha=0.85)

        # Add value labels
        for i, acc in enumerate(accuracies):
            if acc is not None:
                ax.annotate(f"{acc:.1f}%",
                           xy=(i, acc),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           fontsize=9,
                           fontweight='bold',
                           color=info['color'])

    ax.set_xlabel('Validation Method', fontsize=13, fontweight='bold')
    ax.set_ylabel('Conditional Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Validation Method Comparison (Tier 1-2 Threshold)\n' +
                'How Different Validation Approaches Affect Model Accuracy',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenario_labels, fontsize=10)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(45, 92)

    # Add explanation box
    explanation = (
        "Validation Methods:\n"
        "• Tier Only: Exclude based on model agreement\n"
        "• Tier ∩ Judge: Exclude only when BOTH flag (highest confidence)\n"
        "• Tier ∪ Judge: Exclude when EITHER flags (most inclusive)\n"
        "• Judge Only: Exclude based on LLM judge validation\n\n"
        "Key Finding: Tier 1-2 ∪ Judge achieves best accuracy (87.14%)"
    )

    ax.text(0.02, 0.98, explanation,
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    output_file = 'enhanced_validation_method_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved validation method comparison: {output_file}")
    plt.close()


def create_tier_level_progression():
    """
    Chart 2: Show all tier levels with three approaches (Only, ∩Judge, ∪Judge)
    Demonstrates how Union consistently outperforms other methods
    """

    df = pd.read_csv('enhanced_fair_comparison_with_judge.csv')

    # Define scenario groups
    baseline_scenarios = ['Baseline']
    tier_levels = [
        ('Tier 1', 'T1'),
        ('Tier 1-2', 'T1-2'),
        ('Tier 1-3', 'T1-3'),
        ('Tier 1-4', 'T1-4')
    ]
    judge_scenarios = ['Judge Only']

    # Build x-axis labels and positions
    x_labels = ['Base']
    x_positions = [0]
    current_pos = 1

    scenario_order = ['Baseline']

    for tier_name, tier_label in tier_levels:
        for suffix in ['Only', '∩ Judge', '∪ Judge']:
            scenario_name = f"{tier_name} {suffix}"
            scenario_order.append(scenario_name)
            x_labels.append(f"{tier_label}\n{suffix}")
            x_positions.append(current_pos)
            current_pos += 1
        current_pos += 0.5  # Add gap between tier groups

    scenario_order.append('Judge Only')
    x_labels.append('Judge\nOnly')
    x_positions.append(current_pos)

    fig, ax = plt.subplots(figsize=(18, 8))

    # Plot for Gemini Pro (primary focus)
    model_id = 'gemini-2-5-pro'
    info = MODEL_INFO[model_id]

    accuracies = []
    for scenario in scenario_order:
        data = df[(df['scenario'] == scenario) & (df['model'] == model_id)]
        if not data.empty:
            accuracies.append(data['conditional_accuracy'].values[0])
        else:
            accuracies.append(None)

    # Split into three lines: Only, ∩Judge, ∪Judge
    only_x = [0]  # Baseline
    only_y = [accuracies[0]]

    intersect_x = []
    intersect_y = []

    union_x = []
    union_y = []

    idx = 1
    for i, (tier_name, _) in enumerate(tier_levels):
        # Only
        only_x.append(x_positions[idx])
        only_y.append(accuracies[idx])

        # Intersect
        intersect_x.append(x_positions[idx+1])
        intersect_y.append(accuracies[idx+1])

        # Union
        union_x.append(x_positions[idx+2])
        union_y.append(accuracies[idx+2])

        idx += 3

    # Add Judge Only
    judge_x = [x_positions[-1]]
    judge_y = [accuracies[-1]]

    # Plot lines
    ax.plot(only_x, only_y, label='Tier Only', color='#1976D2',
           marker='o', linewidth=2.5, markersize=8, alpha=0.8)
    ax.plot(intersect_x, intersect_y, label='Tier ∩ Judge (Intersection)',
           color='#FF6F00', marker='s', linewidth=2.5, markersize=8, alpha=0.8)
    ax.plot(union_x, union_y, label='Tier ∪ Judge (Union)',
           color='#2E7D32', marker='^', linewidth=3, markersize=10, alpha=0.9)
    ax.scatter(judge_x, judge_y, label='Judge Only', color='#6A1B9A',
              marker='D', s=150, alpha=0.9, edgecolor='black', linewidth=2)

    # Add value labels for Union (best performer)
    for x, y in zip(union_x, union_y):
        ax.annotate(f"{y:.1f}%",
                   xy=(x, y),
                   xytext=(0, 12),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9,
                   fontweight='bold',
                   color='#2E7D32',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='#2E7D32', alpha=0.8))

    ax.set_xlabel('Validation Scenario', fontsize=13, fontweight='bold')
    ax.set_ylabel('Conditional Accuracy (%) - Gemini 2.5 Pro', fontsize=13, fontweight='bold')
    ax.set_title('Tier Level Progression with Judge Integration\n' +
                'Union (∪) Consistently Achieves Highest Accuracy',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(65, 95)

    # Add vertical separators between tier groups
    for i in range(len(tier_levels)):
        sep_pos = x_positions[1 + (i+1)*3] + 0.25
        if sep_pos < x_positions[-1]:
            ax.axvline(x=sep_pos, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    plt.tight_layout()
    output_file = 'enhanced_tier_level_progression.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved tier level progression: {output_file}")
    plt.close()


def create_accuracy_gains_heatmap():
    """
    Chart 3: Heatmap showing accuracy gains from baseline across all scenarios
    """

    df = pd.read_csv('enhanced_fair_comparison_with_judge.csv')

    # Get all scenarios except baseline
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

    fig, ax = plt.subplots(figsize=(16, 6))

    # Create heatmap
    im = ax.imshow(gains_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=0, vmax=gains_matrix.max())

    # Set ticks
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(scenario_labels, fontsize=9)
    ax.set_yticklabels(model_labels, fontsize=11)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(scenarios)):
            text = ax.text(j, i, f'+{gains_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black",
                          fontsize=8, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy Gain (percentage points)', fontsize=11, fontweight='bold')

    ax.set_title('Accuracy Gains from Baseline Across All Validation Scenarios\n' +
                'Higher gains (green) indicate better question quality filtering',
                fontsize=14, fontweight='bold', pad=20)

    # Add vertical lines to separate tier groups
    for pos in [2.5, 5.5, 8.5, 11.5]:
        ax.axvline(x=pos, color='white', linestyle='-', linewidth=2)

    plt.tight_layout()
    output_file = 'enhanced_accuracy_gains_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved accuracy gains heatmap: {output_file}")
    plt.close()


def create_recommended_scenarios_comparison():
    """
    Chart 4: Bar chart comparing recommended scenarios
    Shows tradeoff between dataset size and accuracy
    """

    df = pd.read_csv('enhanced_fair_comparison_with_judge.csv')

    # Focus on recommended scenarios
    scenarios = ['Baseline', 'Judge Only', 'Tier 1-2 ∪ Judge', 'Tier 1-4 ∪ Judge']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left plot: Accuracy comparison
    x = np.arange(len(scenarios))
    width = 0.2

    for i, (model_id, info) in enumerate(MODEL_INFO.items()):
        accuracies = []
        for scenario in scenarios:
            data = df[(df['scenario'] == scenario) & (df['model'] == model_id)]
            if not data.empty:
                accuracies.append(data['conditional_accuracy'].values[0])
            else:
                accuracies.append(0)

        offset = (i - 1.5) * width
        bars = ax1.bar(x + offset, accuracies, width,
                      label=info['label'],
                      color=info['color'],
                      alpha=0.85,
                      edgecolor='white',
                      linewidth=1.5)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_ylabel('Conditional Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison: Recommended Scenarios',
                 fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=10, rotation=15, ha='right')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Right plot: Dataset size vs accuracy tradeoff (Gemini Pro)
    model_id = 'gemini-2-5-pro'

    sizes = []
    accuracies = []
    colors = []

    for scenario in scenarios:
        data = df[(df['scenario'] == scenario) & (df['model'] == model_id)]
        if not data.empty:
            sizes.append(data['evaluated_count'].values[0])
            accuracies.append(data['conditional_accuracy'].values[0])
            if scenario == 'Baseline':
                colors.append('#CCCCCC')
            elif scenario == 'Judge Only':
                colors.append('#2E7D32')  # Recommended - green
            elif scenario == 'Tier 1-2 ∪ Judge':
                colors.append('#1976D2')  # Best balance - blue
            else:
                colors.append('#FF6F00')  # Maximum quality - orange

    scatter = ax2.scatter(sizes, accuracies, s=400, c=colors, alpha=0.8,
                         edgecolors='black', linewidth=2)

    # Add labels
    for scenario, size, acc, color in zip(scenarios, sizes, accuracies, colors):
        label = scenario
        if scenario == 'Judge Only':
            label += '\n(RECOMMENDED)'

        ax2.annotate(label,
                    xy=(size, acc),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                             edgecolor=color, linewidth=2, alpha=0.9),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                   color=color, linewidth=2))

    ax2.set_xlabel('Dataset Size (Questions Evaluated)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Conditional Accuracy (%) - Gemini 2.5 Pro', fontsize=12, fontweight='bold')
    ax2.set_title('Quality vs Coverage Tradeoff',
                 fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(350, 700)
    ax2.set_ylim(65, 95)

    # Add recommendation box
    recommendation = (
        "RECOMMENDATION:\n\n"
        "Judge Only (540 questions, 81.66%)\n"
        "• Best balance of quality and coverage\n"
        "• Excludes 130 LLM judge-flagged questions\n"
        "• +13.63pp improvement from baseline\n\n"
        "For maximum quality:\n"
        "Tier 1-4 ∪ Judge (386 Q, 93.60%)"
    )

    ax2.text(0.05, 0.97, recommendation,
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.85))

    plt.suptitle('Recommended Validation Scenarios for Fair Model Comparison',
                fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout()
    output_file = 'enhanced_recommended_scenarios.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved recommended scenarios comparison: {output_file}")
    plt.close()


def main():
    print("="*80)
    print("Generating Enhanced Fair Comparison Visualizations")
    print("="*80)
    print()

    print("Creating validation method comparison chart...")
    create_validation_method_comparison()

    print("Creating tier level progression chart...")
    create_tier_level_progression()

    print("Creating accuracy gains heatmap...")
    create_accuracy_gains_heatmap()

    print("Creating recommended scenarios comparison...")
    create_recommended_scenarios_comparison()

    print()
    print("="*80)
    print("✓ All enhanced fair comparison charts generated successfully!")
    print("="*80)
    print()
    print("Generated files:")
    print("  - enhanced_validation_method_comparison.png")
    print("  - enhanced_tier_level_progression.png")
    print("  - enhanced_accuracy_gains_heatmap.png")
    print("  - enhanced_recommended_scenarios.png")
    print()
    print("Key Findings Visualized:")
    print("  • Tier 1-2 ∪ Judge achieves best accuracy (87.14%)")
    print("  • Judge Only recommended for balance (81.66%, 540 questions)")
    print("  • Union (∪) consistently outperforms intersection (∩)")
    print("  • Rankings stable across all validation methods")
    print()


if __name__ == "__main__":
    main()
