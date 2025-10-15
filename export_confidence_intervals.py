#!/usr/bin/env python3
"""
Export Confidence Intervals for All Metrics

Calculates and exports Wilson score 95% confidence intervals for all models
and scenarios to a CSV file for easy reference.
"""

import pandas as pd
import math
from pathlib import Path


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


def wilson_score_interval(successes, trials, confidence=0.95):
    """
    Calculate Wilson score confidence interval for binomial proportion.

    Returns:
        tuple: (point_estimate, lower_bound, upper_bound) as percentages
    """
    if trials == 0:
        return (0, 0, 0)

    p = successes / trials
    z = norm_ppf((1 + confidence) / 2)

    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt((p * (1 - p) / trials + z**2 / (4 * trials**2))) / denominator

    lower = max(0, center - margin) * 100
    upper = min(1, center + margin) * 100
    estimate = p * 100

    return (estimate, lower, upper)


def calculate_all_confidence_intervals():
    """Calculate confidence intervals for all metrics and export to CSV."""

    print("="*80)
    print("Calculating Confidence Intervals")
    print("="*80)
    print()

    # Load data
    csv_path = Path("archive/enhanced_fair_comparison_with_judge.csv")
    df = pd.read_csv(csv_path)

    # Filter to key scenarios
    key_scenarios = [
        'Baseline',
        'Judge Only',
        'Tier 1-2 Only',
        'Tier 1-2 ∩ Judge',
        'Tier 1-2 ∪ Judge'
    ]

    df_filtered = df[df['scenario'].isin(key_scenarios)].copy()

    # Calculate CIs for all rows
    results = []

    for _, row in df_filtered.iterrows():
        scenario = row['scenario']
        model = row['model']
        total = row['total_evaluated']
        valid = row['valid_responses']
        correct = row['correct_answers']

        # Response Rate CI
        rr_est, rr_lower, rr_upper = wilson_score_interval(valid, total)

        # Overall Accuracy CI
        acc_est, acc_lower, acc_upper = wilson_score_interval(correct, total)

        # Conditional Accuracy CI
        if valid > 0:
            cond_est, cond_lower, cond_upper = wilson_score_interval(correct, valid)
        else:
            cond_est, cond_lower, cond_upper = (0, 0, 0)

        results.append({
            'scenario': scenario,
            'model': model,
            'n_total': total,
            'n_valid': valid,
            'n_correct': correct,
            'response_rate': f"{rr_est:.2f}%",
            'response_rate_ci': f"[{rr_lower:.2f}%, {rr_upper:.2f}%]",
            'overall_accuracy': f"{acc_est:.2f}%",
            'overall_accuracy_ci': f"[{acc_lower:.2f}%, {acc_upper:.2f}%]",
            'conditional_accuracy': f"{cond_est:.2f}%",
            'conditional_accuracy_ci': f"[{cond_lower:.2f}%, {cond_upper:.2f}%]"
        })

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Export to CSV
    output_path = "confidence_intervals_summary.csv"
    results_df.to_csv(output_path, index=False)
    print(f"✓ Exported to: {output_path}")
    print()

    # Also create a formatted display table for each scenario
    model_labels = {
        'gemini-2-5-pro': 'Gemini 2.5 Pro',
        'gemini-2-5-flash': 'Gemini 2.5 Flash',
        'claude-opus-4-1': 'Claude Opus 4.1',
        'claude-sonnet-4-5': 'Claude Sonnet 4.5'
    }

    print("="*80)
    print("Confidence Intervals by Scenario")
    print("="*80)
    print()

    for scenario in key_scenarios:
        scenario_data = results_df[results_df['scenario'] == scenario]

        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario}")
        print(f"{'='*80}\n")

        for _, row in scenario_data.iterrows():
            model_name = model_labels.get(row['model'], row['model'])
            print(f"{model_name} (n={row['n_total']}):")
            print(f"  Response Rate:       {row['response_rate']:>7} {row['response_rate_ci']}")
            print(f"  Overall Accuracy:    {row['overall_accuracy']:>7} {row['overall_accuracy_ci']}")
            print(f"  Conditional Accuracy: {row['conditional_accuracy']:>7} {row['conditional_accuracy_ci']}")
            print()

    # Create a focused table for conditional accuracy (most important metric)
    print("\n" + "="*80)
    print("CONDITIONAL ACCURACY WITH 95% CONFIDENCE INTERVALS")
    print("="*80)
    print()

    # Pivot table for conditional accuracy
    pivot_data = []
    for scenario in key_scenarios:
        row_data = {'Scenario': scenario}
        scenario_data = results_df[results_df['scenario'] == scenario]
        n = scenario_data.iloc[0]['n_total'] if len(scenario_data) > 0 else 0
        row_data['N'] = n

        for model_id, model_name in model_labels.items():
            model_row = scenario_data[scenario_data['model'] == model_id]
            if len(model_row) > 0:
                row_data[model_name] = f"{model_row.iloc[0]['conditional_accuracy']} {model_row.iloc[0]['conditional_accuracy_ci']}"
            else:
                row_data[model_name] = "N/A"

        pivot_data.append(row_data)

    pivot_df = pd.DataFrame(pivot_data)

    # Print formatted table
    print(pivot_df.to_string(index=False))
    print()

    # Export pivot table
    pivot_output = "conditional_accuracy_with_ci.csv"
    pivot_df.to_csv(pivot_output, index=False)
    print(f"✓ Exported pivot table to: {pivot_output}")
    print()

    print("="*80)
    print("✅ Confidence interval calculation complete!")
    print("="*80)


if __name__ == "__main__":
    calculate_all_confidence_intervals()
