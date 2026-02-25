#!/usr/bin/env python3
"""
Multi-Metric Panel Chart with Statistical Rigor

Creates a 3-panel visualization showing:
- Response Rate | Overall Accuracy | Conditional Accuracy
- Wilson score confidence intervals for all metrics
- Statistical significance markers for pairwise comparisons
- Sample sizes annotated for each scenario

This addresses critical scientific gaps identified in the methodology review:
1. Quantifies uncertainty with proper confidence intervals
2. Shows all three metrics simultaneously (not just conditional accuracy)
3. Makes data exclusion costs visible
4. Enables fair comparison with statistical testing
"""

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from pathlib import Path
import math

matplotlib.use('Agg')
# Use DejaVu Sans for Unicode symbol support (∩, ∪, emoji)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


def norm_ppf(p):
    """
    Inverse of standard normal CDF (percent point function).
    Approximation using Beasley-Springer-Moro algorithm.
    """
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")

    # Coefficients for the approximation
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]

    y = p - 0.5
    if abs(y) < 0.42:
        r = y * y
        x = y * (((a[3]*r + a[2])*r + a[1])*r + a[0]) / \
            ((((b[3]*r + b[2])*r + b[1])*r + b[0])*r + 1)
    else:
        r = p if y > 0 else 1 - p
        r = math.log(-math.log(r))
        x = c[0] + r * (c[1] + r * (c[2] + r * (c[3] + r * (c[4] + r * (c[5] + r * (c[6] + r * (c[7] + r * c[8])))))))
        if y < 0:
            x = -x
    return x


def norm_cdf(x):
    """
    Standard normal cumulative distribution function.
    Uses error function approximation.
    """
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def wilson_score_interval(successes, trials, confidence=0.95):
    """
    Calculate Wilson score confidence interval for binomial proportion.

    The Wilson score interval maintains proper coverage even for extreme
    proportions (near 0% or 100%) and small sample sizes, unlike normal
    approximation which can produce invalid intervals outside [0, 1].

    Args:
        successes: Number of successes
        trials: Total number of trials
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        tuple: (point_estimate, lower_bound, upper_bound)
    """
    if trials == 0:
        return (0, 0, 0)

    p = successes / trials
    z = norm_ppf((1 + confidence) / 2)

    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt((p * (1 - p) / trials + z**2 / (4 * trials**2))) / denominator

    lower = max(0, center - margin)  # Ensure bounds are in [0, 1]
    upper = min(1, center + margin)

    return (p, lower, upper)


def two_proportion_z_test(n1, k1, n2, k2):
    """
    Two-proportion z-test for comparing proportions between groups.

    Tests H0: p1 = p2 vs H1: p1 ≠ p2

    Args:
        n1, k1: sample size and successes for group 1
        n2, k2: sample size and successes for group 2

    Returns:
        float: two-tailed p-value
    """
    if n1 == 0 or n2 == 0:
        return 1.0

    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)

    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

    if se == 0:
        return 1.0

    z = (p1 - p2) / se
    p_value = 2 * (1 - norm_cdf(abs(z)))

    return p_value


def calculate_metrics_with_ci(row, confidence=0.95):
    """
    Calculate all three metrics with Wilson score confidence intervals.

    Args:
        row: DataFrame row with columns: total_evaluated, valid_responses, correct_answers
        confidence: Confidence level for intervals

    Returns:
        dict: Metrics with (estimate, lower_ci, upper_ci, n) for each
    """
    total = row['total_evaluated']
    valid = row['valid_responses']
    correct = row['correct_answers']

    # Response rate: valid / total
    rr_est, rr_lower, rr_upper = wilson_score_interval(valid, total, confidence)

    # Overall accuracy: correct / total
    acc_est, acc_lower, acc_upper = wilson_score_interval(correct, total, confidence)

    # Conditional accuracy: correct / valid (if valid > 0)
    if valid > 0:
        cond_est, cond_lower, cond_upper = wilson_score_interval(correct, valid, confidence)
    else:
        cond_est, cond_lower, cond_upper = (0, 0, 0)

    return {
        'response_rate': {
            'estimate': rr_est * 100,
            'lower': rr_lower * 100,
            'upper': rr_upper * 100,
            'n': total,
            'k': valid
        },
        'overall_accuracy': {
            'estimate': acc_est * 100,
            'lower': acc_lower * 100,
            'upper': acc_upper * 100,
            'n': total,
            'k': correct
        },
        'conditional_accuracy': {
            'estimate': cond_est * 100,
            'lower': cond_lower * 100,
            'upper': cond_upper * 100,
            'n': valid,
            'k': correct
        }
    }


def add_significance_brackets(ax, x_positions, data_list, y_offset=2, fontsize=9):
    """
    Add significance markers for pairwise comparisons between scenarios.

    Compares each scenario to baseline (first scenario) and marks significant
    differences with asterisks and brackets.

    Args:
        ax: Matplotlib axis
        x_positions: X coordinates of bars
        data_list: List of dicts with 'n' and 'k' keys for each scenario
        y_offset: Vertical offset for brackets above bars
        fontsize: Font size for significance markers
    """
    baseline_data = data_list[0]

    for i in range(1, len(data_list)):
        current_data = data_list[i]

        # Two-proportion z-test
        p_value = two_proportion_z_test(
            baseline_data['n'], baseline_data['k'],
            current_data['n'], current_data['k']
        )

        # Determine significance level
        if p_value < 0.001:
            marker = '***'
        elif p_value < 0.01:
            marker = '**'
        elif p_value < 0.05:
            marker = '*'
        else:
            continue  # Not significant

        # Get max height for bracket placement
        max_height = max(
            data_list[0]['estimate'] + (data_list[0]['upper'] - data_list[0]['estimate']),
            data_list[i]['estimate'] + (data_list[i]['upper'] - data_list[i]['estimate'])
        )

        y = max_height + y_offset

        # Draw bracket
        x1, x2 = x_positions[0], x_positions[i]
        ax.plot([x1, x1, x2, x2], [y-0.5, y, y, y-0.5], 'k-', linewidth=1, clip_on=False)

        # Add marker
        ax.text((x1 + x2) / 2, y + 0.5, marker,
               ha='center', va='bottom', fontsize=fontsize,
               fontweight='bold', clip_on=False)


def create_multi_metric_panel():
    """Create 3-panel multi-metric visualization with statistical rigor."""
    print("="*80)
    print("Multi-Metric Panel Chart with Confidence Intervals")
    print("="*80)
    print()

    # Load data
    csv_path = Path("archive/enhanced_fair_comparison_with_judge.csv")
    df = pd.read_csv(csv_path)

    # Model info - enhanced colors and patterns for clarity and accessibility
    # 7 models: 4 Claude + 3 Gemini
    models_info = {
        'claude-opus-4-6': {
            'label': 'Claude Opus 4.6',
            'color': '#8B0000',
            'pattern': None,
            'order': 0
        },
        'claude-opus-4-5': {
            'label': 'Claude Opus 4.5',
            'color': '#C74444',
            'pattern': '///',
            'order': 1
        },
        'claude-sonnet-4-6': {
            'label': 'Claude Sonnet 4.6',
            'color': '#E67878',
            'pattern': None,
            'order': 2
        },
        'claude-sonnet-4-5': {
            'label': 'Claude Sonnet 4.5',
            'color': '#F5A623',
            'pattern': '///',
            'order': 3
        },
        'gemini-3-flash': {
            'label': 'Gemini 3 Flash',
            'color': '#34A853',
            'pattern': None,
            'order': 4
        },
        'gemini-3-pro': {
            'label': 'Gemini 3 Pro',
            'color': '#1A5490',
            'pattern': '///',
            'order': 5
        },
        'gemini-3-1-pro': {
            'label': 'Gemini 3.1 Pro',
            'color': '#1A237E',
            'pattern': None,
            'order': 6
        },
    }

    # Select key scenarios for comparison (as recommended by methodology review)
    # Focus on: Baseline, Judge Only, Tier 1-2 Only, Tier 1-2∩Judge, Tier 1-2∪Judge
    key_scenarios = [
        'Baseline',
        'Judge Only',
        'Tier 1-2 Only',
        'Tier 1-2 ∩ Judge',
        'Tier 1-2 ∪ Judge'
    ]

    df_filtered = df[df['scenario'].isin(key_scenarios)].copy()

    # Calculate metrics with CIs for all rows
    metrics_data = []
    for _, row in df_filtered.iterrows():
        metrics = calculate_metrics_with_ci(row)
        metrics_data.append({
            'scenario': row['scenario'],
            'model': row['model'],
            'evaluated_count': row['evaluated_count'],
            **metrics
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.patch.set_facecolor('white')

    metric_names = ['response_rate', 'overall_accuracy', 'conditional_accuracy']
    metric_titles = ['Response Rate', 'Overall Accuracy', 'Conditional Accuracy']

    # Bar width and positions
    num_models = len(models_info)
    num_scenarios = len(key_scenarios)
    x = np.arange(num_scenarios)
    width = 0.11
    offsets = np.linspace(-width*(num_models-1)/2, width*(num_models-1)/2, num_models)

    # Plot each panel
    for panel_idx, (metric_name, metric_title) in enumerate(zip(metric_names, metric_titles)):
        ax = axes[panel_idx]
        ax.set_facecolor('#FAFAFA')

        # Plot bars for each model
        for model_id, model_info in sorted(models_info.items(), key=lambda x: x[1]['order']):
            model_label = model_info['label']
            color = model_info['color']
            offset = offsets[model_info['order']]

            # Extract data for this model across scenarios
            model_data = metrics_df[metrics_df['model'] == model_id]

            estimates = []
            lower_errors = []
            upper_errors = []
            data_for_sig = []

            for scenario in key_scenarios:
                scenario_data = model_data[model_data['scenario'] == scenario]
                if len(scenario_data) > 0:
                    row = scenario_data.iloc[0]
                    metric_vals = row[metric_name]
                    estimates.append(metric_vals['estimate'])
                    # Ensure errors are positive
                    lower_errors.append(abs(metric_vals['estimate'] - metric_vals['lower']))
                    upper_errors.append(abs(metric_vals['upper'] - metric_vals['estimate']))
                    data_for_sig.append({
                        'estimate': metric_vals['estimate'],
                        'lower': metric_vals['lower'],
                        'upper': metric_vals['upper'],
                        'n': metric_vals['n'],
                        'k': metric_vals['k']
                    })
                else:
                    estimates.append(0)
                    lower_errors.append(0)
                    upper_errors.append(0)
                    data_for_sig.append({'estimate': 0, 'lower': 0, 'upper': 0, 'n': 0, 'k': 0})

            # Get pattern for this model
            pattern = model_info['pattern']

            # Plot bars with patterns and error bars
            bars = ax.bar(x + offset, estimates, width,
                         label=model_label if panel_idx == 0 else "",
                         color=color, alpha=0.9,
                         edgecolor='white', linewidth=1.5,
                         hatch=pattern)

            # Add error bars (Wilson score CIs)
            ax.errorbar(x + offset, estimates,
                       yerr=[lower_errors, upper_errors],
                       fmt='none', ecolor='black', elinewidth=1.5,
                       capsize=3, capthick=1.5, alpha=0.7)

            # Add value labels above bars (no more text inside bars)
            for i, (bar, est) in enumerate(zip(bars, estimates)):
                ax.text(bar.get_x() + bar.get_width()/2, est + upper_errors[i] + 1.5,
                       f'{est:.1f}%',
                       ha='center', va='bottom', fontsize=8,
                       fontweight='bold', color='#333333')

        # Add significance brackets for first model only (to avoid clutter)
        if panel_idx > 0:  # Skip for response rate (all high)
            first_model = sorted(models_info.items(), key=lambda x: x[1]['order'])[0][0]
            model_data = metrics_df[metrics_df['model'] == first_model]

            data_for_sig = []
            for scenario in key_scenarios:
                scenario_data = model_data[model_data['scenario'] == scenario]
                if len(scenario_data) > 0:
                    row = scenario_data.iloc[0]
                    metric_vals = row[metric_name]
                    data_for_sig.append({
                        'estimate': metric_vals['estimate'],
                        'lower': metric_vals['lower'],
                        'upper': metric_vals['upper'],
                        'n': metric_vals['n'],
                        'k': metric_vals['k']
                    })

            if len(data_for_sig) == len(key_scenarios):
                add_significance_brackets(ax, x + offsets[0], data_for_sig,
                                        y_offset=3, fontsize=8)

        # Styling
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold', color='#333333')
        ax.set_title(metric_title, fontsize=14, fontweight='bold',
                    pad=15, color='#1a1a1a')

        # X-axis labels with sample sizes
        x_labels = []
        for scenario in key_scenarios:
            n = metrics_df[metrics_df['scenario'] == scenario]['evaluated_count'].iloc[0]
            x_labels.append(f"{scenario}\n({n}Q)")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9, color='#333333')

        # Set appropriate y-limits
        if metric_name == 'response_rate':
            ax.set_ylim(90, 100)
        else:
            ax.set_ylim(0, 105)

        # Grid and spines
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.8, color='#CCCCCC')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')

    # Add enhanced legend (only to first panel to avoid duplication)
    legend = axes[0].legend(loc='lower left', fontsize=11, framealpha=0.98,
                           edgecolor='#999999', fancybox=False,
                           shadow=False, frameon=True,
                           handlelength=2.5, handleheight=1.2,
                           borderpad=0.8, labelspacing=0.6)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(1.2)

    # Add overall title
    fig.suptitle('Model Performance: Response Rate, Overall Accuracy, and Conditional Accuracy\n'
                'With 95% Wilson Score Confidence Intervals and Statistical Significance Testing',
                fontsize=16, fontweight='bold', y=0.98)

    # Add footer note
    fig.text(0.5, 0.02,
            'Note: Error bars show 95% Wilson score confidence intervals. '
            'Statistical significance markers: * p<0.05, ** p<0.01, *** p<0.001 (vs Baseline, Gemini Pro only for clarity).',
            ha='center', fontsize=9, style='italic', color='#666666')

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    # Save
    output_path = "performance_multi_metric_panel_with_ci.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"✓ Saved: {output_path}")

    # Also save as PDF for publication
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"✓ Saved: {pdf_path}")
    print()

    # Print summary statistics
    print("="*80)
    print("Statistical Summary with 95% Confidence Intervals")
    print("="*80)

    for scenario in key_scenarios:
        print(f"\n{scenario}:")
        scenario_data = metrics_df[metrics_df['scenario'] == scenario]

        for _, row in scenario_data.iterrows():
            model_label = models_info[row['model']]['label']
            print(f"\n  {model_label}:")

            for metric_name, metric_title in zip(metric_names, metric_titles):
                vals = row[metric_name]
                print(f"    {metric_title}: {vals['estimate']:.2f}% "
                      f"[95% CI: {vals['lower']:.2f}%-{vals['upper']:.2f}%] "
                      f"(n={vals['n']})")

    print()
    print("="*80)
    print("✅ Multi-metric panel visualization complete!")
    print("="*80)

    plt.close()


if __name__ == "__main__":
    create_multi_metric_panel()
