#!/usr/bin/env python3
"""
Organize questions by tier for manual review

Creates folder structure under TLUE/model_answer/tier_analysis/
with complete question data organized by tier category.
"""

import json
import csv
import os
from pathlib import Path
from collections import Counter, defaultdict

def load_all_data():
    """Load judge validation and all model answers"""

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
                'judge_answer': val.get('correct_answer', ''),
                'judge_explanation': val.get('answer_explanation', ''),
                'judge_confidence': val.get('confidence', '')
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
                    'answer_key': entry.get('answer', ''),
                    'question': entry.get('polished_ti_content', ''),
                    'choices': entry.get('choices', []),
                    'category': entry.get('category', '')
                }

        all_answers[model_id] = model_data

    return judge_data, all_answers

def categorize_question(gemini_pro, gemini_flash, claude_opus, claude_sonnet, answer_key):
    """Categorize a question based on model agreement pattern (MECE)"""

    answers = [gemini_pro, gemini_flash, claude_opus, claude_sonnet]
    if None in answers:
        return 'extraction_failed', None

    if all(ans == answer_key for ans in answers):
        return 'all_correct', None

    if len(set(answers)) == 1 and answers[0] != answer_key:
        return 'tier1_all_4_agree', answers[0]

    answer_counts = Counter(answers)
    for ans, count in answer_counts.items():
        if count == 3 and ans != answer_key:
            return 'tier2_3_agree', ans

    # Cross-brand
    gemini_answers = {gemini_pro, gemini_flash}
    claude_answers = {claude_opus, claude_sonnet}

    cross_brand_agreements = []
    for g_ans in gemini_answers:
        for c_ans in claude_answers:
            if g_ans == c_ans and g_ans != answer_key:
                cross_brand_agreements.append(g_ans)

    if cross_brand_agreements:
        agreement_counts = Counter(cross_brand_agreements)
        most_common_ans = agreement_counts.most_common(1)[0][0]
        return 'tier3_cross_brand', most_common_ans

    # Tier 4
    gemini_agree = (gemini_pro == gemini_flash and gemini_pro != answer_key)
    claude_agree = (claude_opus == claude_sonnet and claude_opus != answer_key)

    if gemini_agree and claude_agree:
        return 'tier4_both_brands', gemini_pro
    elif gemini_agree:
        return 'tier4_gemini_only', gemini_pro
    elif claude_agree:
        return 'tier4_claude_only', claude_opus

    return 'tier5_no_consensus', None

def create_tier_folders(base_path):
    """Create folder structure for tier analysis"""

    tier_folders = [
        'tier1_all_4_agree',
        'tier2_3_agree',
        'tier3_cross_brand',
        'tier4_both_brands',
        'tier4_gemini_only',
        'tier4_claude_only',
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

    # 2. Save as CSV
    csv_path = tier_path / 'questions.csv'
    if questions_data:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['loc', 'answer_key', 'gemini_pro', 'gemini_flash',
                         'claude_opus', 'claude_sonnet', 'agreed_answer',
                         'judge_flagged', 'judge_answer', 'category', 'question_preview']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for q in questions_data:
                writer.writerow({
                    'loc': q['loc'],
                    'answer_key': q['answer_key'],
                    'gemini_pro': q['gemini_pro'],
                    'gemini_flash': q['gemini_flash'],
                    'claude_opus': q['claude_opus'],
                    'claude_sonnet': q['claude_sonnet'],
                    'agreed_answer': q.get('agreed_answer', ''),
                    'judge_flagged': 'Yes' if q.get('judge_flagged') else 'No',
                    'judge_answer': q.get('judge_answer', ''),
                    'category': q.get('category', ''),
                    'question_preview': q.get('question', '')[:100]
                })

    # 3. Save summary
    summary_path = tier_path / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"="*80 + "\n")
        f.write(f"Tier: {tier_name}\n")
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
            f.write(f"  Gemini Pro:    {q['gemini_pro']}\n")
            f.write(f"  Gemini Flash:  {q['gemini_flash']}\n")
            f.write(f"  Claude Opus:   {q['claude_opus']}\n")
            f.write(f"  Claude Sonnet: {q['claude_sonnet']}\n")
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
    print("="*80)
    print()

    # Load data
    print("Loading data...")
    judge_data, all_answers = load_all_data()
    print(f"✓ Loaded data for {len(all_answers['gemini-2-5-pro'])} questions")
    print()

    # Create folder structure
    base_path = Path("TLUE/model_answer/tier_analysis")
    print(f"Creating folder structure at: {base_path}")
    tier_folders = create_tier_folders(base_path)
    print(f"✓ Created {len(tier_folders)} tier folders")
    print()

    # Categorize and organize questions
    print("Categorizing questions...")
    tier_data = defaultdict(list)

    all_locs = list(all_answers['gemini-2-5-pro'].keys())

    for loc in all_locs:
        # Get model answers
        gemini_pro_data = all_answers['gemini-2-5-pro'][loc]
        gemini_flash_data = all_answers['gemini-2-5-flash'][loc]
        claude_opus_data = all_answers['claude-opus-4-1'][loc]
        claude_sonnet_data = all_answers['claude-sonnet-4-5'][loc]

        gemini_pro = gemini_pro_data['extracted']
        gemini_flash = gemini_flash_data['extracted']
        claude_opus = claude_opus_data['extracted']
        claude_sonnet = claude_sonnet_data['extracted']
        answer_key = gemini_pro_data['answer_key']

        # Categorize
        category, agreed_answer = categorize_question(
            gemini_pro, gemini_flash, claude_opus, claude_sonnet, answer_key
        )

        # Get judge info
        judge_info = judge_data.get(loc, {})
        judge_flagged = judge_info.get('answer_key_correct') == False

        # Compile question data
        question_data = {
            'loc': loc,
            'tier': category,
            'answer_key': answer_key,
            'gemini_pro': gemini_pro,
            'gemini_flash': gemini_flash,
            'claude_opus': claude_opus,
            'claude_sonnet': claude_sonnet,
            'agreed_answer': agreed_answer,
            'question': gemini_pro_data['question'],
            'choices': gemini_pro_data['choices'],
            'category': gemini_pro_data['category'],
            'judge_flagged': judge_flagged,
            'judge_answer': judge_info.get('judge_answer', ''),
            'judge_explanation': judge_info.get('judge_explanation', ''),
            'judge_confidence': judge_info.get('judge_confidence', '')
        }

        tier_data[category].append(question_data)

    print(f"✓ Categorized {len(all_locs)} questions")
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
            print(f"  ✓ {tier_name}: {len(questions)} questions saved")

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
    print("✓ Organization complete!")
    print(f"Data saved to: {base_path}")
    print()
    print("Each tier folder contains:")
    print("  - questions.jsonl  (full data in JSON Lines format)")
    print("  - questions.csv    (spreadsheet format)")
    print("  - summary.txt      (human-readable summary)")
    print("="*80)

if __name__ == "__main__":
    main()
