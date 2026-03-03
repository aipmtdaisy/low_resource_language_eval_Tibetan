#!/usr/bin/env python3
"""
Calculate fair comparison metrics using 5-tier model agreement thresholds.
Excludes questions with likely bad answer keys based on cross-model agreement patterns.

Generalized for N models with brand-based tier logic.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional

# 8 models: new round + previous round (gemini-3-pro & gemini-3-1-pro excluded pending re-run)
MODELS = [
    'claude-opus-4-6', 'claude-opus-4-5', 'claude-sonnet-4-6', 'claude-sonnet-4-5',
    'claude-opus-4-1',
    'gemini-3-flash', 'gemini-2-5-flash', 'gemini-2-5-pro',
]

BRAND_MAP = {
    'claude-opus-4-6': 'claude',
    'claude-opus-4-5': 'claude',
    'claude-sonnet-4-6': 'claude',
    'claude-sonnet-4-5': 'claude',
    'claude-opus-4-1': 'claude',
    'gemini-3-flash': 'gemini',
    'gemini-2-5-flash': 'gemini',
    'gemini-2-5-pro': 'gemini',
}

NUM_MODELS = len(MODELS)


def load_all_model_answers():
    """Load answers from all models"""
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
                    'answer_key': entry.get('answer', ''),
                    'question': entry.get('polished_ti_content', '')[:200]
                }

        all_answers[model_id] = model_data

    return all_answers


def categorize_question(model_answers: Dict[str, Optional[str]], answer_key: str) -> Tuple[str, Optional[str]]:
    """Categorize a question based on model agreement pattern.

    Tier logic for 7 models:
        Tier 1: 6+ models agree on answer != key
        Tier 2: 5 models agree on answer != key
        Tier 3: 3-4 agree with cross-brand agreement (>=1 Gemini + >=1 Claude)
        Tier 4: 2+ agree, same-brand only
        Tier 5: No consensus
    """
    answers = list(model_answers.values())

    if None in answers:
        return 'extraction_failed', None

    if all(ans == answer_key for ans in answers):
        return 'all_correct', None

    answer_counts = Counter(answers)
    non_key_counts = {ans: cnt for ans, cnt in answer_counts.items() if ans != answer_key}
    if not non_key_counts:
        return 'all_correct', None

    top_answer, top_count = max(non_key_counts.items(), key=lambda x: x[1])

    # Tier 1: near-unanimous (6+ of 7)
    if top_count >= NUM_MODELS - 1:
        return 'tier1_near_unanimous', top_answer

    # Tier 2: strong agreement (5 of 7)
    if top_count >= NUM_MODELS - 2:
        return 'tier2_strong_agree', top_answer

    # Check brand composition for remaining tiers
    if top_count >= 3:
        agreeing_brands = set()
        for model_id, ans in model_answers.items():
            if ans == top_answer:
                agreeing_brands.add(BRAND_MAP[model_id])
        if len(agreeing_brands) >= 2:
            return 'tier3_cross_brand', top_answer
        else:
            return 'tier4_same_brand', top_answer

    # Check 2-model agreements
    for ans, cnt in non_key_counts.items():
        if cnt >= 2:
            agreeing_brands = set()
            for model_id, model_ans in model_answers.items():
                if model_ans == ans:
                    agreeing_brands.add(BRAND_MAP[model_id])
            if len(agreeing_brands) >= 2:
                return 'tier3_cross_brand', ans
            else:
                return 'tier4_same_brand', ans

    return 'tier5_no_consensus', None


def get_excluded_locs_for_threshold(categories: Dict, threshold: str) -> Set[str]:
    """Get set of question locs to exclude based on threshold"""
    excluded = set()

    tier_keys_by_level = {
        'tier1': ['tier1_near_unanimous'],
        'tier1-2': ['tier1_near_unanimous', 'tier2_strong_agree'],
        'tier1-3': ['tier1_near_unanimous', 'tier2_strong_agree', 'tier3_cross_brand'],
        'tier1-4': ['tier1_near_unanimous', 'tier2_strong_agree', 'tier3_cross_brand', 'tier4_same_brand'],
    }

    keys = tier_keys_by_level.get(threshold, [])
    for key in keys:
        for item in categories.get(key, []):
            excluded.add(item['loc'])

    return excluded


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
    print("Tier-Based Fair Comparison Metrics")
    print(f"Models: {NUM_MODELS} ({', '.join(MODELS)})")
    print("="*120)
    print()
    print("Calculating fair model comparisons using 5-tier agreement thresholds")
    print()

    # Load all model answers
    all_answers = load_all_model_answers()

    # Get all question locs (from first model)
    first_model = MODELS[0]
    all_locs = list(all_answers[first_model].keys())

    # Categorize all questions
    categories = defaultdict(list)

    for loc in all_locs:
        model_ans = {}
        for model_id in MODELS:
            if loc in all_answers[model_id]:
                model_ans[model_id] = all_answers[model_id][loc]['extracted']
            else:
                model_ans[model_id] = None

        answer_key = all_answers[first_model][loc]['answer_key']
        question = all_answers[first_model][loc]['question']

        category, agreed_answer = categorize_question(model_ans, answer_key)

        entry = {
            'loc': loc,
            'question': question,
            'answer_key': answer_key,
            'agreed_answer': agreed_answer
        }
        for model_id in MODELS:
            entry[model_id] = model_ans[model_id]

        categories[category].append(entry)

    # Print tier summary
    total = len(all_locs)
    tier1 = len(categories['tier1_near_unanimous'])
    tier2 = len(categories['tier2_strong_agree'])
    tier3 = len(categories['tier3_cross_brand'])
    tier4 = len(categories['tier4_same_brand'])
    tier5 = len(categories['tier5_no_consensus'])
    correct = len(categories['all_correct'])
    failed = len(categories['extraction_failed'])

    print("Question Categorization Summary:")
    print("-"*120)
    print(f"Total questions: {total}")
    print(f"  Tier 1 ({NUM_MODELS-1}+ agree):         {tier1:3d} ({tier1/total*100:5.1f}%)")
    print(f"  Tier 2 ({NUM_MODELS-2}+ agree):         {tier2:3d} ({tier2/total*100:5.1f}%)")
    print(f"  Tier 3 (cross-brand):          {tier3:3d} ({tier3/total*100:5.1f}%)")
    print(f"  Tier 4 (same-brand):           {tier4:3d} ({tier4/total*100:5.1f}%)")
    print(f"  Tier 5 (no consensus):         {tier5:3d} ({tier5/total*100:5.1f}%)")
    print(f"  All correct:                   {correct:3d} ({correct/total*100:5.1f}%)")
    print(f"  Extraction failed:             {failed:3d} ({failed/total*100:5.1f}%)")
    print()

    # Thresholds to test
    thresholds = [
        ('baseline', 'Baseline (All Questions)', set()),
        ('tier1', 'Conservative (Exclude Tier 1)', None),
        ('tier1-2', 'Balanced (Exclude Tier 1-2)', None),
        ('tier1-3', 'Inclusive (Exclude Tier 1-3)', None),
        ('tier1-4', 'Maximum (Exclude Tier 1-4)', None)
    ]

    # Store all results
    all_results = []

    for threshold_id, threshold_name, excluded_locs in thresholds:
        print("="*120)
        print(f"{threshold_name}")
        print("="*120)

        if excluded_locs is None:
            excluded_locs = get_excluded_locs_for_threshold(categories, threshold_id)

        excluded_count = len(excluded_locs)
        evaluated_count = total - excluded_count

        print(f"Questions excluded: {excluded_count}")
        print(f"Questions evaluated: {evaluated_count}")
        print()

        print(f"{'Model':<25} {'Evaluated':<12} {'Valid Resp':<12} {'Correct':<10} {'Resp Rate':<12} {'Accuracy':<12} {'Cond. Acc':<12}")
        print("-"*120)

        threshold_results = []
        for model in MODELS:
            metrics = calculate_model_metrics(model, all_answers, excluded_locs)
            threshold_results.append(metrics)

            print(f"{metrics['model']:<25} {metrics['total_evaluated']:<12} {metrics['valid_responses']:<12} "
                  f"{metrics['correct_answers']:<10} {metrics['response_rate']:>10.2f}% {metrics['accuracy']:>10.2f}% "
                  f"{metrics['conditional_accuracy']:>10.2f}%")

            metrics['threshold'] = threshold_id
            metrics['threshold_name'] = threshold_name
            metrics['questions_excluded'] = excluded_count
            all_results.append(metrics)

        print()

        # Show ranking by conditional accuracy
        ranked = sorted(threshold_results, key=lambda x: x['conditional_accuracy'], reverse=True)
        print("Ranking by Conditional Accuracy:")
        for i, r in enumerate(ranked, 1):
            print(f"  {i}. {r['model']:<25} {r['conditional_accuracy']:>6.2f}%")
        print()

    # Save to CSV
    output_file = "tier_based_fair_comparison_metrics.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'threshold', 'threshold_name', 'questions_excluded', 'model',
            'total_evaluated', 'valid_responses', 'correct_answers',
            'response_rate', 'accuracy', 'conditional_accuracy'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print("="*120)
    print(f"Results saved to: {output_file}")
    print("="*120)
    print()

    # Print comparison summary
    print("="*120)
    print("Summary: How Rankings Change Across Thresholds")
    print("="*120)
    print()

    for model in MODELS:
        model_results = [r for r in all_results if r['model'] == model]
        print(f"\n{model}:")
        print(f"{'Threshold':<30} {'Questions':<12} {'Resp Rate':<12} {'Accuracy':<12} {'Cond. Acc':<12}")
        print("-"*90)
        for r in model_results:
            print(f"{r['threshold_name']:<30} {r['total_evaluated']:<12} "
                  f"{r['response_rate']:>10.2f}% {r['accuracy']:>10.2f}% "
                  f"{r['conditional_accuracy']:>10.2f}%")

if __name__ == "__main__":
    main()
