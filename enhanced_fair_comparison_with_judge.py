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

Generalized for N models with brand-based tier logic.
"""

import json
import csv
from pathlib import Path
from collections import Counter, defaultdict
from typing import Set, Dict, List, Optional, Tuple

# New round: 7 models (4 Claude via Bedrock + 3 Gemini)
MODELS = [
    'claude-opus-4-6', 'claude-opus-4-5', 'claude-sonnet-4-6', 'claude-sonnet-4-5',
    'gemini-3-flash', 'gemini-3-pro', 'gemini-3-1-pro'
]

BRAND_MAP = {
    'claude-opus-4-6': 'claude',
    'claude-opus-4-5': 'claude',
    'claude-sonnet-4-6': 'claude',
    'claude-sonnet-4-5': 'claude',
    'gemini-3-flash': 'gemini',
    'gemini-3-pro': 'gemini',
    'gemini-3-1-pro': 'gemini',
}

NUM_MODELS = len(MODELS)

# Which model's judge validation file to use (update if changing judge)
JUDGE_MODEL = 'gemini-3-pro'


def load_all_data():
    """Load judge validation and model answers"""

    # Load judge validation
    validation_file = f"TLUE/model_answer/{JUDGE_MODEL}_eval_res/{JUDGE_MODEL}_llm_validated_v3_retry.jsonl"
    judge_data = {}

    judge_path = Path(validation_file)
    if judge_path.exists():
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
    else:
        print(f"Warning: Judge validation file not found: {validation_file}")
        print("  Judge-based scenarios will use empty dispute set.")

    # Load model answers
    all_answers = {}

    for model_id in MODELS:
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


def categorize_question(model_answers: Dict[str, Optional[str]], answer_key: str) -> str:
    """Categorize a question based on model agreement pattern (MECE)"""
    answers = list(model_answers.values())

    if None in answers:
        return 'extraction_failed'

    if all(ans == answer_key for ans in answers):
        return 'all_correct'

    answer_counts = Counter(answers)
    non_key_counts = {ans: cnt for ans, cnt in answer_counts.items() if ans != answer_key}
    if not non_key_counts:
        return 'all_correct'

    top_answer, top_count = max(non_key_counts.items(), key=lambda x: x[1])

    if top_count >= NUM_MODELS - 1:
        return 'tier1_near_unanimous'

    if top_count >= NUM_MODELS - 2:
        return 'tier2_strong_agree'

    if top_count >= 3:
        agreeing_brands = set()
        for model_id, ans in model_answers.items():
            if ans == top_answer:
                agreeing_brands.add(BRAND_MAP[model_id])
        if len(agreeing_brands) >= 2:
            return 'tier3_cross_brand'
        else:
            return 'tier4_same_brand'

    for ans, cnt in non_key_counts.items():
        if cnt >= 2:
            agreeing_brands = set()
            for model_id, model_ans in model_answers.items():
                if model_ans == ans:
                    agreeing_brands.add(BRAND_MAP[model_id])
            if len(agreeing_brands) >= 2:
                return 'tier3_cross_brand'
            else:
                return 'tier4_same_brand'

    return 'tier5_no_consensus'


def get_tier_questions(tier_categories: Dict, tier_level: str) -> Set[str]:
    """Get set of question locs for a given tier level"""
    tier_keys_by_level = {
        'tier1': ['tier1_near_unanimous'],
        'tier1-2': ['tier1_near_unanimous', 'tier2_strong_agree'],
        'tier1-3': ['tier1_near_unanimous', 'tier2_strong_agree', 'tier3_cross_brand'],
        'tier1-4': ['tier1_near_unanimous', 'tier2_strong_agree', 'tier3_cross_brand', 'tier4_same_brand'],
    }

    tier_questions = set()
    for key in tier_keys_by_level.get(tier_level, []):
        tier_questions.update(tier_categories.get(key, []))

    return tier_questions


def calculate_model_metrics(model_id: str, all_answers: Dict, excluded_locs: Set[str]) -> Dict:
    """Calculate metrics for a model excluding specified questions"""
    model_data = all_answers[model_id]

    total_evaluated = 0
    valid_responses = 0
    correct_answers = 0

    for loc, data in model_data.items():
        if loc in excluded_locs:
            continue

        total_evaluated += 1
        extracted = data['extracted']
        answer_key = data['answer_key']

        if extracted is not None:
            valid_responses += 1
            if extracted == answer_key:
                correct_answers += 1

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
    print(f"Models: {NUM_MODELS} ({', '.join(MODELS)})")
    print("="*120)
    print()

    # Load data
    print("Loading data...")
    judge_data, all_answers = load_all_data()
    print(f"  Loaded {len(judge_data)} judge validations")
    print()

    # Categorize questions into tiers
    print("Categorizing questions...")
    tier_categories = defaultdict(list)
    first_model = MODELS[0]
    all_locs = list(all_answers[first_model].keys())

    for loc in all_locs:
        model_ans = {}
        for model_id in MODELS:
            if loc in all_answers[model_id]:
                model_ans[model_id] = all_answers[model_id][loc]['extracted']
            else:
                model_ans[model_id] = None

        answer_key = all_answers[first_model][loc]['answer_key']
        category = categorize_question(model_ans, answer_key)
        tier_categories[category].append(loc)

    # Get judge-disputed questions
    judge_disputed = set([loc for loc, info in judge_data.items()
                         if info['answer_key_correct'] == False])

    print(f"  Categorized {len(all_locs)} questions")
    print(f"  Judge disputed: {len(judge_disputed)} questions")
    print()

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

        scenarios.append({
            'name': f'{tier_name} Only',
            'description': f'Exclude {tier_name} ({len(tier_questions)} questions)',
            'excluded': tier_questions.copy()
        })

        tier_and_judge = tier_questions & judge_disputed
        scenarios.append({
            'name': f'{tier_name} \u2229 Judge',
            'description': f'Exclude {tier_name} AND Judge agree ({len(tier_and_judge)} questions)',
            'excluded': tier_and_judge.copy()
        })

        tier_or_judge = tier_questions | judge_disputed
        scenarios.append({
            'name': f'{tier_name} \u222a Judge',
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

        for model in MODELS:
            metrics = calculate_model_metrics(model, all_answers, scenario['excluded'])
            metrics['scenario'] = scenario['name']
            metrics['excluded_count'] = excluded_count
            metrics['evaluated_count'] = evaluated_count
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

    print("Tier 1-2 Comparison (Balanced threshold):")
    print("-"*120)

    tier12_scenarios = [
        'Baseline',
        'Tier 1-2 Only',
        'Tier 1-2 \u2229 Judge',
        'Tier 1-2 \u222a Judge',
        'Judge Only'
    ]

    for model in MODELS:
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

if __name__ == "__main__":
    main()
