#!/usr/bin/env python3
"""
Calculate fair comparison metrics using 5-tier model agreement thresholds.
Excludes questions with likely bad answer keys based on cross-model agreement patterns.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set

def load_all_model_answers():
    """Load answers from all 4 models"""
    models = ['gemini-2-5-pro', 'gemini-2-5-flash', 'claude-opus-4-1', 'claude-sonnet-4-5']

    all_answers = {}  # model_id → (loc → {extracted_answer, answer_key, question})

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
                    'answer_key': entry.get('answer', ''),
                    'question': entry.get('polished_ti_content', '')[:200]
                }

        all_answers[model_id] = model_data

    return all_answers

def categorize_question(gemini_pro, gemini_flash, claude_opus, claude_sonnet, answer_key):
    """Categorize a question based on model agreement pattern"""

    # Skip if any model didn't extract an answer
    answers = [gemini_pro, gemini_flash, claude_opus, claude_sonnet]
    if None in answers:
        return 'extraction_failed', None

    # Check if all agree with key
    if all(ans == answer_key for ans in answers):
        return 'all_correct', None

    # Get unique answers that differ from key
    different_answers = [ans for ans in answers if ans != answer_key]
    if not different_answers:
        return 'all_correct', None

    # Tier 1: All 4 models agree on same answer (different from key)
    if len(set(answers)) == 1 and answers[0] != answer_key:
        return 'tier1_all_4_agree', answers[0]

    # Tier 2: 3 models agree on same answer (different from key)
    answer_counts = Counter(answers)
    for ans, count in answer_counts.items():
        if count == 3 and ans != answer_key:
            return 'tier2_3_agree', ans

    # Tier 3: Cross-brand agreement (1 Gemini + 1 Claude agree, different from key)
    gemini_answers = {gemini_pro, gemini_flash}
    claude_answers = {claude_opus, claude_sonnet}

    cross_brand_agreements = []
    for g_ans in gemini_answers:
        for c_ans in claude_answers:
            if g_ans == c_ans and g_ans != answer_key:
                cross_brand_agreements.append(g_ans)

    if cross_brand_agreements:
        # Find the most common cross-brand agreement
        agreement_counts = Counter(cross_brand_agreements)
        most_common_ans = agreement_counts.most_common(1)[0][0]
        return 'tier3_cross_brand', most_common_ans

    # Tier 4: Same-brand agreement (both Gemini or both Claude agree, different from key)
    gemini_agree = (gemini_pro == gemini_flash and gemini_pro != answer_key)
    claude_agree = (claude_opus == claude_sonnet and claude_opus != answer_key)

    if gemini_agree and claude_agree:
        # Both brands agree (on different answers)
        return 'tier4_both_brands', gemini_pro
    elif gemini_agree:
        return 'tier4_gemini_only', gemini_pro
    elif claude_agree:
        return 'tier4_claude_only', claude_opus

    # Tier 5: No clear agreement
    return 'tier5_no_consensus', None

def get_excluded_locs_for_threshold(categories: Dict, threshold: str) -> Set[str]:
    """Get set of question locs to exclude based on threshold"""
    excluded = set()

    if threshold == 'tier1':
        # Conservative: exclude only tier 1
        for item in categories['tier1_all_4_agree']:
            excluded.add(item['loc'])

    elif threshold == 'tier1-2':
        # Balanced: exclude tier 1-2
        for item in categories['tier1_all_4_agree'] + categories['tier2_3_agree']:
            excluded.add(item['loc'])

    elif threshold == 'tier1-3':
        # Inclusive: exclude tier 1-3
        for item in (categories['tier1_all_4_agree'] +
                     categories['tier2_3_agree'] +
                     categories['tier3_cross_brand']):
            excluded.add(item['loc'])

    elif threshold == 'tier1-4':
        # Maximum: exclude tier 1-4
        for item in (categories['tier1_all_4_agree'] +
                     categories['tier2_3_agree'] +
                     categories['tier3_cross_brand'] +
                     categories['tier4_both_brands'] +
                     categories['tier4_gemini_only'] +
                     categories['tier4_claude_only']):
            excluded.add(item['loc'])

    return excluded

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
    print("Tier-Based Fair Comparison Metrics")
    print("="*120)
    print()
    print("Calculating fair model comparisons using 5-tier agreement thresholds")
    print()

    # Load all model answers
    all_answers = load_all_model_answers()

    # Get all question locs (from first model)
    all_locs = list(all_answers['gemini-2-5-pro'].keys())

    # Categorize all questions
    categories = {
        'tier1_all_4_agree': [],
        'tier2_3_agree': [],
        'tier3_cross_brand': [],
        'tier4_both_brands': [],
        'tier4_gemini_only': [],
        'tier4_claude_only': [],
        'tier5_no_consensus': [],
        'all_correct': [],
        'extraction_failed': []
    }

    for loc in all_locs:
        gemini_pro = all_answers['gemini-2-5-pro'][loc]['extracted']
        gemini_flash = all_answers['gemini-2-5-flash'][loc]['extracted']
        claude_opus = all_answers['claude-opus-4-1'][loc]['extracted']
        claude_sonnet = all_answers['claude-sonnet-4-5'][loc]['extracted']
        answer_key = all_answers['gemini-2-5-pro'][loc]['answer_key']
        question = all_answers['gemini-2-5-pro'][loc]['question']

        category, agreed_answer = categorize_question(
            gemini_pro, gemini_flash, claude_opus, claude_sonnet, answer_key
        )

        categories[category].append({
            'loc': loc,
            'question': question,
            'answer_key': answer_key,
            'gemini_pro': gemini_pro,
            'gemini_flash': gemini_flash,
            'claude_opus': claude_opus,
            'claude_sonnet': claude_sonnet,
            'agreed_answer': agreed_answer
        })

    # Print tier summary
    total = len(all_locs)
    tier1 = len(categories['tier1_all_4_agree'])
    tier2 = len(categories['tier2_3_agree'])
    tier3 = len(categories['tier3_cross_brand'])
    tier4_both = len(categories['tier4_both_brands'])
    tier4_g = len(categories['tier4_gemini_only'])
    tier4_c = len(categories['tier4_claude_only'])
    tier5 = len(categories['tier5_no_consensus'])
    correct = len(categories['all_correct'])
    failed = len(categories['extraction_failed'])

    print("Question Categorization Summary:")
    print("-"*120)
    print(f"Total questions: {total}")
    print(f"  Tier 1 (all 4 agree):          {tier1:3d} ({tier1/total*100:5.1f}%)")
    print(f"  Tier 2 (3 agree):              {tier2:3d} ({tier2/total*100:5.1f}%)")
    print(f"  Tier 3 (cross-brand):          {tier3:3d} ({tier3/total*100:5.1f}%)")
    print(f"  Tier 4 (same-brand):           {tier4_both + tier4_g + tier4_c:3d} ({(tier4_both + tier4_g + tier4_c)/total*100:5.1f}%)")
    print(f"    - Both brands (diff answers):  {tier4_both:3d}")
    print(f"    - Gemini pair only:            {tier4_g:3d}")
    print(f"    - Claude pair only:            {tier4_c:3d}")
    print(f"  Tier 5 (no consensus):         {tier5:3d} ({tier5/total*100:5.1f}%)")
    print(f"  All correct:                   {correct:3d} ({correct/total*100:5.1f}%)")
    print(f"  Extraction failed:             {failed:3d} ({failed/total*100:5.1f}%)")
    print()

    # Models to evaluate
    models = ['gemini-2-5-pro', 'gemini-2-5-flash', 'claude-opus-4-1', 'claude-sonnet-4-5']

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

        # Get excluded locs for this threshold
        if excluded_locs is None:
            excluded_locs = get_excluded_locs_for_threshold(categories, threshold_id)

        excluded_count = len(excluded_locs)
        evaluated_count = total - excluded_count

        print(f"Questions excluded: {excluded_count}")
        print(f"Questions evaluated: {evaluated_count}")
        print()

        # Calculate metrics for each model
        print(f"{'Model':<25} {'Evaluated':<12} {'Valid Resp':<12} {'Correct':<10} {'Resp Rate':<12} {'Accuracy':<12} {'Cond. Acc':<12}")
        print("-"*120)

        threshold_results = []
        for model in models:
            metrics = calculate_model_metrics(model, all_answers, excluded_locs)
            threshold_results.append(metrics)

            print(f"{metrics['model']:<25} {metrics['total_evaluated']:<12} {metrics['valid_responses']:<12} "
                  f"{metrics['correct_answers']:<10} {metrics['response_rate']:>10.2f}% {metrics['accuracy']:>10.2f}% "
                  f"{metrics['conditional_accuracy']:>10.2f}%")

            # Add threshold info for CSV
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

    for model in models:
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
