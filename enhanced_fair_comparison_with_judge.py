#!/usr/bin/env python3
"""
Enhanced Fair Comparison: Tier-Based and Judge-Based Validation Integration

Compares 17 scenarios:
1. Baseline
2-4. Tier 1: Tier only / Tier∩Judge / Tier∪Judge
5-7. Tier 1-2: Tier only / Tier∩Judge / Tier∪Judge
8-10. Tier 1-3: Tier only / Tier∩Judge / Tier∪Judge
11-13. Tier 1-4: Tier only / Tier∩Judge / Tier∪Judge
14. Judge only

Shows which validation method provides best accuracy improvements.
"""

import json
import csv
from pathlib import Path
from collections import Counter, defaultdict
from typing import Set, Dict, List

def load_all_data():
    """Load judge validation and model answers"""

    # Load judge validation
    validation_file = "TLUE/model_answer/gemini-2-5-pro_eval_res/gemini-2-5-pro_llm_validated_v3_retry.jsonl"
    judge_data = {}

    with open(validation_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            loc = entry.get('loc', '')
            val = entry.get('llm_validation', {})
            judge_data[loc] = {
                'answer_key_correct': val.get('answer_key_correct'),
                'judge_answer': val.get('correct_answer', '')
            }

    # Load model answers
    models = ['gemini-2-5-pro', 'gemini-2-5-flash', 'claude-opus-4-1', 'claude-sonnet-4-5']
    all_answers = {}

    for model_id in models:
        combined_file = Path(f"TLUE/model_answer/{model_id}_eval_res/{model_id}_combined_results.jsonl")
        model_data = {}

        with open(combined_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                loc = entry.get('loc', '')
                extracted_list = entry.get('extracted_answer', [])

                model_data[loc] = {
                    'extracted': extracted_list[0] if extracted_list else None,
                    'answer_key': entry.get('answer', '')
                }

        all_answers[model_id] = model_data

    return judge_data, all_answers

def categorize_question(gemini_pro, gemini_flash, claude_opus, claude_sonnet, answer_key):
    """Categorize a question based on model agreement pattern (MECE)"""

    answers = [gemini_pro, gemini_flash, claude_opus, claude_sonnet]
    if None in answers:
        return 'extraction_failed'

    if all(ans == answer_key for ans in answers):
        return 'all_correct'

    if len(set(answers)) == 1 and answers[0] != answer_key:
        return 'tier1_all_4_agree'

    answer_counts = Counter(answers)
    for ans, count in answer_counts.items():
        if count == 3 and ans != answer_key:
            return 'tier2_3_agree'

    # Cross-brand
    gemini_answers = {gemini_pro, gemini_flash}
    claude_answers = {claude_opus, claude_sonnet}

    cross_brand_agreements = []
    for g_ans in gemini_answers:
        for c_ans in claude_answers:
            if g_ans == c_ans and g_ans != answer_key:
                cross_brand_agreements.append(g_ans)

    if cross_brand_agreements:
        return 'tier3_cross_brand'

    # Tier 4
    gemini_agree = (gemini_pro == gemini_flash and gemini_pro != answer_key)
    claude_agree = (claude_opus == claude_sonnet and claude_opus != answer_key)

    if gemini_agree and claude_agree:
        return 'tier4_both_brands'
    elif gemini_agree:
        return 'tier4_gemini_only'
    elif claude_agree:
        return 'tier4_claude_only'

    return 'tier5_no_consensus'

def get_tier_questions(tier_categories: Dict, tier_level: str) -> Set[str]:
    """Get set of question locs for a given tier level"""

    tier_questions = set()

    if tier_level == 'tier1':
        tier_questions.update(tier_categories['tier1_all_4_agree'])
    elif tier_level == 'tier1-2':
        tier_questions.update(tier_categories['tier1_all_4_agree'])
        tier_questions.update(tier_categories['tier2_3_agree'])
    elif tier_level == 'tier1-3':
        tier_questions.update(tier_categories['tier1_all_4_agree'])
        tier_questions.update(tier_categories['tier2_3_agree'])
        tier_questions.update(tier_categories['tier3_cross_brand'])
    elif tier_level == 'tier1-4':
        tier_questions.update(tier_categories['tier1_all_4_agree'])
        tier_questions.update(tier_categories['tier2_3_agree'])
        tier_questions.update(tier_categories['tier3_cross_brand'])
        tier_questions.update(tier_categories['tier4_both_brands'])
        tier_questions.update(tier_categories['tier4_gemini_only'])
        tier_questions.update(tier_categories['tier4_claude_only'])

    return tier_questions

def calculate_model_metrics(model_id: str, all_answers: Dict, excluded_locs: Set[str]) -> Dict:
    """Calculate metrics for a model excluding specified questions"""

    model_data = all_answers[model_id]

    total_evaluated = 0
    valid_responses = 0
    correct_answers = 0

    for loc, data in model_data.items():
        # Skip excluded questions
        if loc in excluded_locs:
            continue

        total_evaluated += 1
        extracted = data['extracted']
        answer_key = data['answer_key']

        # Check if model provided a valid answer
        if extracted is not None:
            valid_responses += 1
            if extracted == answer_key:
                correct_answers += 1

    # Calculate metrics
    response_rate = (valid_responses / total_evaluated * 100) if total_evaluated > 0 else 0
    accuracy = (correct_answers / total_evaluated * 100) if total_evaluated > 0 else 0
    conditional_accuracy = (correct_answers / valid_responses * 100) if valid_responses > 0 else 0

    return {
        'model': model_id,
        'total_evaluated': total_evaluated,
        'valid_responses': valid_responses,
        'correct_answers': correct_answers,
        'response_rate': response_rate,
        'accuracy': accuracy,
        'conditional_accuracy': conditional_accuracy
    }

def main():
    print("="*120)
    print("Enhanced Fair Comparison: Tier-Based and Judge-Based Validation Integration")
    print("="*120)
    print()

    # Load data
    print("Loading data...")
    judge_data, all_answers = load_all_data()
    print("✓ Data loaded")
    print()

    # Categorize questions into tiers
    print("Categorizing questions...")
    tier_categories = defaultdict(list)
    all_locs = list(all_answers['gemini-2-5-pro'].keys())

    for loc in all_locs:
        gemini_pro = all_answers['gemini-2-5-pro'][loc]['extracted']
        gemini_flash = all_answers['gemini-2-5-flash'][loc]['extracted']
        claude_opus = all_answers['claude-opus-4-1'][loc]['extracted']
        claude_sonnet = all_answers['claude-sonnet-4-5'][loc]['extracted']
        answer_key = all_answers['gemini-2-5-pro'][loc]['answer_key']

        category = categorize_question(gemini_pro, gemini_flash, claude_opus, claude_sonnet, answer_key)
        tier_categories[category].append(loc)

    # Get judge-disputed questions
    judge_disputed = set([loc for loc, info in judge_data.items()
                         if info['answer_key_correct'] == False])

    print(f"✓ Categorized {len(all_locs)} questions")
    print(f"  Judge disputed: {len(judge_disputed)} questions")
    print()

    # Models to evaluate
    models = ['gemini-2-5-pro', 'gemini-2-5-flash', 'claude-opus-4-1', 'claude-sonnet-4-5']

    # Define all scenarios
    scenarios = []

    # Baseline
    scenarios.append({
        'name': 'Baseline',
        'description': 'All questions',
        'excluded': set()
    })

    # For each tier level
    tier_levels = [
        ('tier1', 'Tier 1'),
        ('tier1-2', 'Tier 1-2'),
        ('tier1-3', 'Tier 1-3'),
        ('tier1-4', 'Tier 1-4')
    ]

    for tier_id, tier_name in tier_levels:
        tier_questions = get_tier_questions(tier_categories, tier_id)

        # Tier only
        scenarios.append({
            'name': f'{tier_name} Only',
            'description': f'Exclude {tier_name} ({len(tier_questions)} questions)',
            'excluded': tier_questions.copy()
        })

        # Tier ∩ Judge
        tier_and_judge = tier_questions & judge_disputed
        scenarios.append({
            'name': f'{tier_name} ∩ Judge',
            'description': f'Exclude {tier_name} AND Judge agree ({len(tier_and_judge)} questions)',
            'excluded': tier_and_judge.copy()
        })

        # Tier ∪ Judge
        tier_or_judge = tier_questions | judge_disputed
        scenarios.append({
            'name': f'{tier_name} ∪ Judge',
            'description': f'Exclude {tier_name} OR Judge flagged ({len(tier_or_judge)} questions)',
            'excluded': tier_or_judge.copy()
        })

    # Judge only
    scenarios.append({
        'name': 'Judge Only',
        'description': f'Exclude Judge disputed ({len(judge_disputed)} questions)',
        'excluded': judge_disputed.copy()
    })

    print(f"Total scenarios: {len(scenarios)}")
    print()

    # Run all scenarios
    all_results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] {scenario['name']: <25} - {scenario['description']}")

        excluded_count = len(scenario['excluded'])
        evaluated_count = len(all_locs) - excluded_count

        scenario_results = []
        for model in models:
            metrics = calculate_model_metrics(model, all_answers, scenario['excluded'])
            metrics['scenario'] = scenario['name']
            metrics['excluded_count'] = excluded_count
            metrics['evaluated_count'] = evaluated_count
            scenario_results.append(metrics)
            all_results.append(metrics)

    print()
    print("="*120)
    print("Comprehensive Results")
    print("="*120)
    print()

    # Group by scenario
    for scenario in scenarios:
        scenario_name = scenario['name']
        scenario_results = [r for r in all_results if r['scenario'] == scenario_name]

        if not scenario_results:
            continue

        print(f"\n{scenario['name']}")
        print(f"{scenario['description']}")
        print("-"*120)
        print(f"{'Model':<25} {'Evaluated':<12} {'Valid':<10} {'Correct':<10} {'Resp Rate':<12} {'Accuracy':<12} {'Cond. Acc':<12}")
        print("-"*120)

        for r in scenario_results:
            print(f"{r['model']:<25} {r['total_evaluated']:<12} {r['valid_responses']:<10} "
                  f"{r['correct_answers']:<10} {r['response_rate']:>10.2f}% {r['accuracy']:>10.2f}% "
                  f"{r['conditional_accuracy']:>10.2f}%")

    # Save to CSV
    output_file = "enhanced_fair_comparison_with_judge.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'scenario', 'excluded_count', 'evaluated_count', 'model',
            'total_evaluated', 'valid_responses', 'correct_answers',
            'response_rate', 'accuracy', 'conditional_accuracy'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print()
    print("="*120)
    print(f"Results saved to: {output_file}")
    print("="*120)
    print()

    # Comparative analysis
    print("="*120)
    print("Comparative Analysis: Which Validation Method Works Best?")
    print("="*120)
    print()

    # Compare Tier 1-2 approaches
    print("Tier 1-2 Comparison (Balanced threshold):")
    print("-"*120)

    tier12_scenarios = [
        'Baseline',
        'Tier 1-2 Only',
        'Tier 1-2 ∩ Judge',
        'Tier 1-2 ∪ Judge',
        'Judge Only'
    ]

    for model in models:
        print(f"\n{model}:")
        print(f"{'Scenario':<25} {'Questions':<12} {'Cond. Acc':<12} {'Gain from Baseline':<20}")
        print("-"*120)

        baseline_acc = None
        for scenario_name in tier12_scenarios:
            result = next((r for r in all_results if r['scenario'] == scenario_name and r['model'] == model), None)
            if result:
                if scenario_name == 'Baseline':
                    baseline_acc = result['conditional_accuracy']
                    gain = 0.0
                else:
                    gain = result['conditional_accuracy'] - baseline_acc if baseline_acc else 0

                print(f"{scenario_name:<25} {result['evaluated_count']:<12} {result['conditional_accuracy']:>10.2f}% "
                      f"{f'+{gain:.2f}pp' if gain >= 0 else f'{gain:.2f}pp':>18}")

    print()
    print("="*120)
    print("Recommendations")
    print("="*120)
    print()

    # Find best performing scenarios
    claude_opus_results = [r for r in all_results if r['model'] == 'claude-opus-4-1']
    gemini_pro_results = [r for r in all_results if r['model'] == 'gemini-2-5-pro']

    best_claude = max(claude_opus_results, key=lambda x: x['conditional_accuracy'])
    best_gemini = max(gemini_pro_results, key=lambda x: x['conditional_accuracy'])

    print(f"Best scenario for Claude Opus: {best_claude['scenario']} ({best_claude['conditional_accuracy']:.2f}%)")
    print(f"Best scenario for Gemini Pro: {best_gemini['scenario']} ({best_gemini['conditional_accuracy']:.2f}%)")
    print()

    # Check if Tier ∩ Judge gives highest confidence
    tier12_intersect = next((r for r in all_results
                            if r['scenario'] == 'Tier 1-2 ∩ Judge' and r['model'] == 'gemini-2-5-pro'), None)
    tier12_only = next((r for r in all_results
                       if r['scenario'] == 'Tier 1-2 Only' and r['model'] == 'gemini-2-5-pro'), None)
    judge_only = next((r for r in all_results
                      if r['scenario'] == 'Judge Only' and r['model'] == 'gemini-2-5-pro'), None)

    if tier12_intersect and tier12_only and judge_only:
        print("Validation Method Comparison:")
        print(f"  Tier 1-2 Only:     {tier12_only['conditional_accuracy']:.2f}% ({tier12_only['evaluated_count']} questions)")
        print(f"  Judge Only:        {judge_only['conditional_accuracy']:.2f}% ({judge_only['evaluated_count']} questions)")
        print(f"  Tier 1-2 ∩ Judge:  {tier12_intersect['conditional_accuracy']:.2f}% ({tier12_intersect['evaluated_count']} questions)")
        print()

        if tier12_intersect['conditional_accuracy'] > max(tier12_only['conditional_accuracy'], judge_only['conditional_accuracy']):
            print("✓ RECOMMENDATION: Use Tier 1-2 ∩ Judge (highest confidence, both methods agree)")
            print(f"  This excludes {tier12_intersect['excluded_count']} questions where both validation methods flagged issues")
        elif tier12_only['conditional_accuracy'] > judge_only['conditional_accuracy']:
            print("✓ RECOMMENDATION: Use Tier 1-2 Only (tier-based validation more reliable)")
            print(f"  This excludes {tier12_only['excluded_count']} questions based on cross-model agreement")
        else:
            print("✓ RECOMMENDATION: Use Judge Only (judge validation more reliable)")
            print(f"  This excludes {judge_only['excluded_count']} questions flagged by LLM judge")

    print()
    print("="*120)

if __name__ == "__main__":
    main()
