#!/usr/bin/env python3
"""
Create radar charts comparing model performance across validation scenarios
Shows how different validation methods (Tier-based, Judge-based, and combinations)
affect model accuracy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Model configuration (7 models: 4 Claude + 3 Gemini)
MODEL_INFO = {
    'claude-opus-4-6': {'color': '#8B0000', 'label': 'Claude Opus 4.6'},
    'claude-opus-4-5': {'color': '#EA4335', 'label': 'Claude Opus 4.5'},
    'claude-sonnet-4-6': {'color': '#E67878', 'label': 'Claude Sonnet 4.6'},
    'claude-sonnet-4-5': {'color': '#FBBC04', 'label': 'Claude Sonnet 4.5'},
    'gemini-3-flash': {'color': '#34A853', 'label': 'Gemini 3 Flash'},
    'gemini-3-pro': {'color': '#4285F4', 'label': 'Gemini 3 Pro'},
    'gemini-3-1-pro': {'color': '#1A237E', 'label': 'Gemini 3.1 Pro'},
}


def create_individual_model_radar():
    """Create separate radar chart for each model showing their performance across scenarios"""

    df = pd.read_csv('enhanced_fair_comparison_with_judge.csv')

    # Focus on Tier 1-2 comparison scenarios (balanced threshold)
    scenarios = [
        'Baseline',
        'Tier 1-2 Only',
        'Tier 1-2 ∩ Judge',
        'Tier 1-2 ∪ Judge',
        'Judge Only'
    ]

    scenario_labels = [
        'Baseline\n(670 Q)',
        'Tier 1-2\nOnly',
        'Tier 1-2 ∩\nJudge',
        'Tier 1-2 ∪\nJudge',
        'Judge\nOnly'
    ]

    num_vars = len(scenarios)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(3, 3, figsize=(22, 22), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()

    for idx, (model_id, info) in enumerate(MODEL_INFO.items()):
        ax = axes[idx]

        # Get conditional accuracy for this model across scenarios
        values = []
        for scenario in scenarios:
            data = df[(df['scenario'] == scenario) & (df['model'] == model_id)]
            if not data.empty:
                values.append(data['conditional_accuracy'].values[0])
            else:
                values.append(0)

        values += values[:1]  # Close the plot

        # Plot
        ax.plot(angles, values, 'o-', linewidth=3,
               color=info['color'], markersize=10)
        ax.fill(angles, values, alpha=0.25, color=info['color'])

        # Add value labels
        for angle, value in zip(angles[:-1], values[:-1]):
            x = angle
            y = value
            ax.text(x, y + 3, f'{value:.1f}%',
                   ha='center', va='center', fontsize=9,
                   fontweight='bold', color=info['color'],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=info['color'], alpha=0.9))

        # Configure axes
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(scenario_labels, fontsize=10, fontweight='bold')

        # Set y-axis
        baseline_val = values[0]
        max_val = max(values[:-1])
        y_min = max(0, baseline_val - 10)
        y_max = min(100, max_val + 10)

        ax.set_ylim(y_min, y_max)
        y_ticks = np.linspace(y_min, y_max, 5)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:.0f}%' for y in y_ticks], fontsize=9)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

        # Title
        ax.set_title(f'{info["label"]}\nBest: {max(values[:-1]):.1f}% '
                    f'(+{max(values[:-1]) - baseline_val:.1f}pp)',
                    fontsize=13, fontweight='bold', pad=20, color=info['color'])

    # Hide unused subplots
    for idx in range(len(MODEL_INFO), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Model Performance Across Validation Scenarios\n'
                 'Tier 1-2 Threshold Comparison',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()
    output_file = 'enhanced_radar_individual_models.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved individual model radar charts: {output_file}")
    plt.close()


def create_combined_model_radar():
    """Create single radar chart with all 4 models for comparison"""

    df = pd.read_csv('enhanced_fair_comparison_with_judge.csv')

    scenarios = [
        'Baseline',
        'Tier 1-2 Only',
        'Tier 1-2 ∩ Judge',
        'Tier 1-2 ∪ Judge',
        'Judge Only'
    ]

    scenario_labels = [
        'Baseline',
        'Tier 1-2\nOnly',
        'Tier 1-2 ∩\nJudge',
        'Tier 1-2 ∪\nJudge',
        'Judge\nOnly'
    ]

    num_vars = len(scenarios)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))

    # Plot each model
    for model_id, info in MODEL_INFO.items():
        values = []
        for scenario in scenarios:
            data = df[(df['scenario'] == scenario) & (df['model'] == model_id)]
            if not data.empty:
                values.append(data['conditional_accuracy'].values[0])
            else:
                values.append(0)

        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=3,
               label=info['label'], color=info['color'], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=info['color'])

    # Configure axes
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(scenario_labels, fontsize=13, fontweight='bold')

    ax.set_ylim(45, 90)
    ax.set_yticks([50, 60, 70, 80, 90])
    ax.set_yticklabels(['50%', '60%', '70%', '80%', '90%'], fontsize=11)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
             fontsize=13, framealpha=0.95)

    plt.title('All Models: Performance Across Validation Scenarios\n'
             'Tier 1-2 Threshold - Union (∪) Achieves Best Results',
             fontsize=16, fontweight='bold', pad=30, y=1.08)

    # Add annotation
    annotation = (
        "Key Finding:\n"
        "• Union (∪) consistently achieves highest accuracy\n"
        "• Excludes questions flagged by EITHER validation method\n"
        "• Tier 1-2 ∪ Judge: 493 questions, best performance"
    )

    plt.text(0.02, 0.98, annotation,
            transform=plt.gcf().transFigure,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    output_file = 'enhanced_radar_combined_models.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined model radar chart: {output_file}")
    plt.close()


def create_tier_progression_radar():
    """Create radar charts showing progression across tier levels for best model"""

    df = pd.read_csv('enhanced_fair_comparison_with_judge.csv')

    model_id = 'gemini-3-pro'  # Focus on best performing model
    info = MODEL_INFO[model_id]

    # Show progression for each tier level
    tier_groups = [
        ('Tier 1', ['Baseline', 'Tier 1 Only', 'Tier 1 ∩ Judge', 'Tier 1 ∪ Judge']),
        ('Tier 1-2', ['Baseline', 'Tier 1-2 Only', 'Tier 1-2 ∩ Judge', 'Tier 1-2 ∪ Judge']),
        ('Tier 1-3', ['Baseline', 'Tier 1-3 Only', 'Tier 1-3 ∩ Judge', 'Tier 1-3 ∪ Judge']),
        ('Tier 1-4', ['Baseline', 'Tier 1-4 Only', 'Tier 1-4 ∩ Judge', 'Tier 1-4 ∪ Judge'])
    ]

    scenario_labels = ['Baseline', 'Tier\nOnly', 'Tier ∩\nJudge', 'Tier ∪\nJudge']

    fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()

    for idx, (tier_name, scenarios) in enumerate(tier_groups):
        ax = axes[idx]

        num_vars = len(scenarios)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        values = []
        for scenario in scenarios:
            data = df[(df['scenario'] == scenario) & (df['model'] == model_id)]
            if not data.empty:
                values.append(data['conditional_accuracy'].values[0])
            else:
                values.append(0)

        values += values[:1]

        # Plot
        ax.plot(angles, values, 'o-', linewidth=3,
               color=info['color'], markersize=10)
        ax.fill(angles, values, alpha=0.25, color=info['color'])

        # Add value labels
        for angle, value in zip(angles[:-1], values[:-1]):
            ax.text(angle, value + 2, f'{value:.1f}%',
                   ha='center', va='center', fontsize=10,
                   fontweight='bold', color=info['color'],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=info['color'], alpha=0.9))

        # Configure axes
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(scenario_labels, fontsize=11, fontweight='bold')

        baseline_val = values[0]
        max_val = max(values[:-1])

        ax.set_ylim(baseline_val - 5, max_val + 5)
        y_ticks = np.linspace(baseline_val - 5, max_val + 5, 5)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:.0f}%' for y in y_ticks], fontsize=9)

        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

        # Get question counts
        union_scenario = scenarios[-1]
        union_data = df[(df['scenario'] == union_scenario) & (df['model'] == model_id)]
        if not union_data.empty:
            q_count = union_data['evaluated_count'].values[0]
            excluded = union_data['excluded_count'].values[0]
        else:
            q_count = 0
            excluded = 0

        ax.set_title(f'{tier_name}\n'
                    f'Best: {max_val:.1f}% (+{max_val - baseline_val:.1f}pp)\n'
                    f'{q_count} questions (excluded {excluded})',
                    fontsize=12, fontweight='bold', pad=20, color=info['color'])

    plt.suptitle(f'{info["label"]}: Tier Level Progression\n'
                 'Union (∪) Consistently Achieves Highest Accuracy',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()
    output_file = 'enhanced_radar_tier_progression.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved tier progression radar chart: {output_file}")
    plt.close()


def create_metric_comparison_radar():
    """Create radar comparing Response Rate, Accuracy, and Conditional Accuracy"""

    df = pd.read_csv('enhanced_fair_comparison_with_judge.csv')

    # Compare key scenarios
    scenarios = ['Baseline', 'Judge Only', 'Tier 1-2 ∪ Judge']
    scenario_labels = ['Baseline\n(670 Q)', 'Judge Only\n(540 Q)', 'Tier 1-2 ∪ Judge\n(493 Q)']

    num_vars = len(scenarios)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), subplot_kw=dict(projection='polar'))

    metrics = [
        ('response_rate', 'Response Rate', 90, 100),
        ('accuracy', 'Overall Accuracy', 45, 90),
        ('conditional_accuracy', 'Conditional Accuracy', 45, 95)
    ]

    for idx, (metric_key, metric_label, y_min, y_max) in enumerate(metrics):
        ax = axes[idx]

        for model_id, info in MODEL_INFO.items():
            values = []
            for scenario in scenarios:
                data = df[(df['scenario'] == scenario) & (df['model'] == model_id)]
                if not data.empty:
                    values.append(data[metric_key].values[0])
                else:
                    values.append(0)

            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2.5,
                   label=info['label'], color=info['color'], markersize=8)
            ax.fill(angles, values, alpha=0.15, color=info['color'])

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(scenario_labels, fontsize=10, fontweight='bold')

        ax.set_ylim(y_min, y_max)
        y_ticks = np.linspace(y_min, y_max, 5)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:.0f}%' for y in y_ticks], fontsize=9)

        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax.set_title(metric_label, fontsize=13, fontweight='bold', pad=20)

        if idx == 2:
            ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1.1),
                     fontsize=11, framealpha=0.95)

    plt.suptitle('Metric Comparison Across Validation Scenarios',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    output_file = 'enhanced_radar_metric_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved metric comparison radar chart: {output_file}")
    plt.close()


def main():
    print("="*80)
    print("Generating Enhanced Radar Comparison Charts")
    print("="*80)
    print()

    print("Creating individual model radar charts...")
    create_individual_model_radar()

    print("Creating combined model radar chart...")
    create_combined_model_radar()

    print("Creating tier progression radar chart...")
    create_tier_progression_radar()

    print("Creating metric comparison radar chart...")
    create_metric_comparison_radar()

    print()
    print("="*80)
    print("✓ All enhanced radar charts generated successfully!")
    print("="*80)
    print()
    print("Generated files:")
    print("  - enhanced_radar_individual_models.png")
    print("  - enhanced_radar_combined_models.png")
    print("  - enhanced_radar_tier_progression.png")
    print("  - enhanced_radar_metric_comparison.png")
    print()


if __name__ == "__main__":
    main()
