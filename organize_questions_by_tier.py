#!/usr/bin/env python3
"""
Organize questions by tier for manual review

Creates folder structure under TLUE/model_answer/tier_analysis/
with complete question data organized by tier category.

Generalized for N models with brand-based tier logic.
"""

import json
import csv
import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Optional, Tuple

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

# Which model's judge validation file to use
JUDGE_MODEL = 'gemini-2-5-pro'


def load_all_data():
    """Load judge validation and all model answers"""

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
                    'judge_answer': val.get('correct_answer', ''),
                    'judge_explanation': val.get('answer_explanation', ''),
                    'judge_confidence': val.get('confidence', '')
                }
    else:
        print(f"Warning: Judge validation file not found: {validation_file}")
        print("  Judge data will be empty.")

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
                    'answer_key': entry.get('answer', ''),
                    'question': entry.get('polished_ti_content', ''),
                    'choices': entry.get('choices', []),
                    'category': entry.get('category', '')
                }

        all_answers[model_id] = model_data

    return judge_data, all_answers


def categorize_question(model_answers: Dict[str, Optional[str]], answer_key: str) -> Tuple[str, Optional[str]]:
    """Categorize a question based on model agreement pattern (MECE)"""
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

    # Tier 1: near-unanimous (NUM_MODELS-1 or more agree)
    if top_count >= NUM_MODELS - 1:
        return 'tier1_near_unanimous', top_answer

    # Tier 2: strong agreement (NUM_MODELS-2 agree)
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


def create_tier_folders(base_path):
    """Create folder structure for tier analysis"""

    tier_folders = [
        'tier1_near_unanimous',
        'tier2_strong_agree',
        'tier3_cross_brand',
        'tier4_same_brand',
        'tier5_no_consensus',
        'all_correct',
        'extraction_failed'
    ]

    for tier in tier_folders:
        folder_path = base_path / tier
        folder_path.mkdir(parents=True, exist_ok=True)

    return tier_folders

def save_tier_data(tier_name, questions_data, base_path, judge_data):
    """Save questions for a specific tier in multiple formats"""

    tier_path = base_path / tier_name

    # 1. Save as JSONL
    jsonl_path = tier_path / 'questions.jsonl'
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for q in questions_data:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')

    # 2. Save as CSV with dynamic model columns
    csv_path = tier_path / 'questions.csv'
    if questions_data:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['loc', 'answer_key']
            # Add per-model columns dynamically
            for model_id in MODELS:
                fieldnames.append(model_id)
            fieldnames.extend(['agreed_answer', 'judge_flagged', 'judge_answer',
                             'category', 'question_preview'])
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for q in questions_data:
                row = {
                    'loc': q['loc'],
                    'answer_key': q['answer_key'],
                    'agreed_answer': q.get('agreed_answer', ''),
                    'judge_flagged': 'Yes' if q.get('judge_flagged') else 'No',
                    'judge_answer': q.get('judge_answer', ''),
                    'category': q.get('category', ''),
                    'question_preview': q.get('question', '')[:100]
                }
                for model_id in MODELS:
                    row[model_id] = q.get(model_id, '')
                writer.writerow(row)

    # 3. Save summary
    summary_path = tier_path / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"="*80 + "\n")
        f.write(f"Tier: {tier_name}\n")
        f.write(f"Models: {NUM_MODELS} ({', '.join(MODELS)})\n")
        f.write(f"="*80 + "\n\n")

        f.write(f"Total questions: {len(questions_data)}\n\n")

        # Judge statistics
        judge_flagged = sum(1 for q in questions_data if q.get('judge_flagged'))
        if len(questions_data) > 0:
            f.write(f"Judge flagged as incorrect: {judge_flagged} ({judge_flagged/len(questions_data)*100:.1f}%)\n\n")

        # Category breakdown
        categories = {}
        for q in questions_data:
            cat = q.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        if categories:
            f.write("Questions by category:\n")
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                f.write(f"  {cat}: {count}\n")
            f.write("\n")

        # Show first 5 examples
        f.write("="*80 + "\n")
        f.write("Examples (first 5 questions):\n")
        f.write("="*80 + "\n\n")

        for i, q in enumerate(questions_data[:5], 1):
            f.write(f"Example {i}: {q['loc']}\n")
            f.write(f"  Category: {q.get('category', 'N/A')}\n")
            f.write(f"  Answer Key: {q['answer_key']}\n")
            for model_id in MODELS:
                f.write(f"  {model_id:<22} {q.get(model_id, 'N/A')}\n")
            if q.get('agreed_answer'):
                f.write(f"  Models agree on: {q['agreed_answer']}\n")
            if q.get('judge_flagged'):
                f.write(f"  Judge flagged: YES\n")
                f.write(f"  Judge says correct answer: {q.get('judge_answer', 'N/A')}\n")
            f.write(f"  Question: {q.get('question', '')[:200]}...\n")
            f.write("\n")

def main():
    print("="*80)
    print("Organizing Questions by Tier for Manual Review")
    print(f"Models: {NUM_MODELS} ({', '.join(MODELS)})")
    print("="*80)
    print()

    # Load data
    print("Loading data...")
    judge_data, all_answers = load_all_data()
    first_model = MODELS[0]
    print(f"  Loaded data for {len(all_answers[first_model])} questions")
    print()

    # Create folder structure
    base_path = Path("TLUE/model_answer/tier_analysis")
    print(f"Creating folder structure at: {base_path}")
    tier_folders = create_tier_folders(base_path)
    print(f"  Created {len(tier_folders)} tier folders")
    print()

    # Categorize and organize questions
    print("Categorizing questions...")
    tier_data = defaultdict(list)

    all_locs = list(all_answers[first_model].keys())

    for loc in all_locs:
        # Build model_answers dict for this question
        model_ans = {}
        for model_id in MODELS:
            if loc in all_answers[model_id]:
                model_ans[model_id] = all_answers[model_id][loc]['extracted']
            else:
                model_ans[model_id] = None

        answer_key = all_answers[first_model][loc]['answer_key']
        first_model_data = all_answers[first_model][loc]

        # Categorize
        category, agreed_answer = categorize_question(model_ans, answer_key)

        # Get judge info
        judge_info = judge_data.get(loc, {})
        judge_flagged = judge_info.get('answer_key_correct') == False

        # Compile question data
        question_data = {
            'loc': loc,
            'tier': category,
            'answer_key': answer_key,
            'agreed_answer': agreed_answer,
            'question': first_model_data['question'],
            'choices': first_model_data['choices'],
            'category': first_model_data['category'],
            'judge_flagged': judge_flagged,
            'judge_answer': judge_info.get('judge_answer', ''),
            'judge_explanation': judge_info.get('judge_explanation', ''),
            'judge_confidence': judge_info.get('judge_confidence', '')
        }
        # Add per-model answers
        for model_id in MODELS:
            question_data[model_id] = model_ans[model_id]

        tier_data[category].append(question_data)

    print(f"  Categorized {len(all_locs)} questions")
    print()

    # Save data for each tier
    print("Saving tier data...")
    tier_stats = []

    for tier_name in tier_folders:
        questions = tier_data[tier_name]
        if questions:
            save_tier_data(tier_name, questions, base_path, judge_data)
            judge_flagged = sum(1 for q in questions if q['judge_flagged'])
            tier_stats.append({
                'tier': tier_name,
                'count': len(questions),
                'judge_flagged': judge_flagged,
                'judge_rate': judge_flagged / len(questions) * 100 if len(questions) > 0 else 0
            })
            print(f"  {tier_name}: {len(questions)} questions saved")

    print()
    print("="*80)
    print("Summary")
    print("="*80)
    print()

    print(f"{'Tier':<30} {'Questions':<12} {'Judge Flagged':<15} {'Judge Rate':<12}")
    print("-"*80)
    for stat in tier_stats:
        print(f"{stat['tier']:<30} {stat['count']:<12} {stat['judge_flagged']:<15} {stat['judge_rate']:>10.1f}%")

    print()
    print("="*80)
    print("Organization complete!")
    print(f"Data saved to: {base_path}")
    print()
    print("Each tier folder contains:")
    print("  - questions.jsonl  (full data in JSON Lines format)")
    print("  - questions.csv    (spreadsheet format)")
    print("  - summary.txt      (human-readable summary)")
    print("="*80)

if __name__ == "__main__":
    main()
