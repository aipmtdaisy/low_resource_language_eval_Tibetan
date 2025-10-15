#!/usr/bin/env python3
"""
Enhanced Judge Bias Analysis: Combining LLM Judge Validation with Tier-Based Model Agreement

Analyzes:
1. Overlap between judge-disputed questions and tier-based bad keys
2. Over-identification: Judge flagged but tiers say OK
3. Under-identification: Tiers flagged but judge missed
4. Family bias: Does judge favor Gemini over Claude?
5. Bias strength across different evidence tiers
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

def load_judge_validation():
    """Load judge's validation results"""
    validation_file = "TLUE/model_answer/gemini-2-5-pro_eval_res/gemini-2-5-pro_llm_validated_v3_retry.jsonl"

    judge_data = {}  # loc → {answer_key_correct, judge_answer, explanation}

    with open(validation_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            entry = json.loads(line)
            loc = entry.get('loc', '')
            val = entry.get('llm_validation', {})

            judge_data[loc] = {
                'answer_key': entry.get('answer', ''),
                'answer_key_correct': val.get('answer_key_correct'),
                'judge_answer': val.get('correct_answer', ''),
                'explanation': val.get('answer_explanation', '')[:200],
                'question': entry.get('polished_ti_content', '')[:150]
            }

    return judge_data

def load_all_model_answers():
    """Load answers from all 4 models"""
    models = ['gemini-2-5-pro', 'gemini-2-5-flash', 'claude-opus-4-1', 'claude-sonnet-4-5']

    all_answers = {}  # model_id → (loc → {extracted_answer, answer_key})

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

def analyze_family_bias(judge_data, all_answers):
    """Analyze if judge shows family bias when models disagree"""

    bias_stats = {
        'total_disputed': 0,
        'judge_matches_gemini_only': 0,
        'judge_matches_claude_only': 0,
        'judge_matches_both': 0,
        'judge_matches_neither': 0
    }

    bias_examples = {
        'gemini_only': [],
        'claude_only': []
    }

    for loc, judge_info in judge_data.items():
        if judge_info['answer_key_correct'] != False:
            continue

        bias_stats['total_disputed'] += 1

        judge_ans = judge_info['judge_answer']

        gemini_pro = all_answers['gemini-2-5-pro'][loc]['extracted']
        gemini_flash = all_answers['gemini-2-5-flash'][loc]['extracted']
        claude_opus = all_answers['claude-opus-4-1'][loc]['extracted']
        claude_sonnet = all_answers['claude-sonnet-4-5'][loc]['extracted']

        matches_gemini = judge_ans in [gemini_pro, gemini_flash]
        matches_claude = judge_ans in [claude_opus, claude_sonnet]

        if matches_gemini and matches_claude:
            bias_stats['judge_matches_both'] += 1
        elif matches_gemini:
            bias_stats['judge_matches_gemini_only'] += 1
            if len(bias_examples['gemini_only']) < 3:
                bias_examples['gemini_only'].append({
                    'loc': loc,
                    'answer_key': judge_info['answer_key'],
                    'judge': judge_ans,
                    'gemini_pro': gemini_pro,
                    'gemini_flash': gemini_flash,
                    'claude_opus': claude_opus,
                    'claude_sonnet': claude_sonnet,
                    'explanation': judge_info['explanation']
                })
        elif matches_claude:
            bias_stats['judge_matches_claude_only'] += 1
            if len(bias_examples['claude_only']) < 3:
                bias_examples['claude_only'].append({
                    'loc': loc,
                    'answer_key': judge_info['answer_key'],
                    'judge': judge_ans,
                    'gemini_pro': gemini_pro,
                    'gemini_flash': gemini_flash,
                    'claude_opus': claude_opus,
                    'claude_sonnet': claude_sonnet,
                    'explanation': judge_info['explanation']
                })
        else:
            bias_stats['judge_matches_neither'] += 1

    return bias_stats, bias_examples

def main():
    print("="*100)
    print("Enhanced Judge Bias Analysis: Judge Validation vs Tier-Based Model Agreement")
    print("="*100)
    print()

    # Load data
    print("Loading data...")
    judge_data = load_judge_validation()
    all_answers = load_all_model_answers()

    # Get all locs
    all_locs = list(all_answers['gemini-2-5-pro'].keys())

    # Categorize all questions into tiers
    print("Categorizing questions into tiers...")
    tier_categories = defaultdict(list)

    for loc in all_locs:
        gemini_pro = all_answers['gemini-2-5-pro'][loc]['extracted']
        gemini_flash = all_answers['gemini-2-5-flash'][loc]['extracted']
        claude_opus = all_answers['claude-opus-4-1'][loc]['extracted']
        claude_sonnet = all_answers['claude-sonnet-4-5'][loc]['extracted']
        answer_key = all_answers['gemini-2-5-pro'][loc]['answer_key']

        category, agreed_answer = categorize_question(
            gemini_pro, gemini_flash, claude_opus, claude_sonnet, answer_key
        )

        tier_categories[category].append(loc)

    # Extract judge disputed questions
    judge_disputed = set([loc for loc, info in judge_data.items()
                         if info['answer_key_correct'] == False])

    # Tier-based bad keys (Tier 1-4)
    tier_bad_keys = set()
    for tier in ['tier1_all_4_agree', 'tier2_3_agree', 'tier3_cross_brand',
                 'tier4_both_brands', 'tier4_gemini_only', 'tier4_claude_only']:
        tier_bad_keys.update(tier_categories[tier])

    # All correct questions
    all_correct = set(tier_categories['all_correct'])

    print()
    print("="*100)
    print("Data Summary")
    print("="*100)
    print()
    print(f"Total questions: {len(all_locs)}")
    print(f"Judge disputed (marked as incorrect): {len(judge_disputed)}")
    print(f"Tier-based bad keys (Tier 1-4): {len(tier_bad_keys)}")
    print(f"  - Tier 1 (all 4 agree): {len(tier_categories['tier1_all_4_agree'])}")
    print(f"  - Tier 2 (3 agree): {len(tier_categories['tier2_3_agree'])}")
    print(f"  - Tier 3 (cross-brand): {len(tier_categories['tier3_cross_brand'])}")
    print(f"  - Tier 4 (same-brand): {len(tier_categories['tier4_both_brands']) + len(tier_categories['tier4_gemini_only']) + len(tier_categories['tier4_claude_only'])}")
    print(f"    - Both brands (diff answers): {len(tier_categories['tier4_both_brands'])}")
    print(f"    - Gemini pair only: {len(tier_categories['tier4_gemini_only'])}")
    print(f"    - Claude pair only: {len(tier_categories['tier4_claude_only'])}")
    print(f"All correct (all 4 match key): {len(all_correct)}")
    print()

    # Cross-reference analysis
    print("="*100)
    print("Cross-Reference Analysis: Judge vs Tiers")
    print("="*100)
    print()

    # 1. Agreement (both flagged)
    agreement = judge_disputed & tier_bad_keys

    # 2. Judge over-identification (judge flagged, tiers say OK)
    over_id = judge_disputed - tier_bad_keys

    # 3. Judge under-identification (tiers flagged, judge missed)
    under_id = tier_bad_keys - judge_disputed

    print(f"✓ AGREEMENT (both judge and tiers flagged):     {len(agreement):3d} questions")
    print(f"⚠ OVER-IDENTIFICATION (judge flagged, tiers OK): {len(over_id):3d} questions")
    print(f"⚠ UNDER-IDENTIFICATION (tiers flagged, missed):  {len(under_id):3d} questions")
    print()

    # Calculate precision and recall
    precision = len(agreement) / len(judge_disputed) * 100 if len(judge_disputed) > 0 else 0
    recall = len(agreement) / len(tier_bad_keys) * 100 if len(tier_bad_keys) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("="*100)
    print("Judge Performance Metrics")
    print("="*100)
    print()
    print(f"Precision: {precision:.1f}% ({len(agreement)}/{len(judge_disputed)})")
    print(f"  - Of questions judge flagged, how many are validated by tier agreement")
    print()
    print(f"Recall: {recall:.1f}% ({len(agreement)}/{len(tier_bad_keys)})")
    print(f"  - Of tier-validated bad keys, how many did judge catch")
    print()
    print(f"F1 Score: {f1:.1f}%")
    print()

    # Breakdown of agreement by tier
    print("="*100)
    print("Agreement Breakdown by Tier (Judge ∩ Tiers)")
    print("="*100)
    print()

    tier_breakdown = {}
    for tier_name in ['tier1_all_4_agree', 'tier2_3_agree', 'tier3_cross_brand',
                      'tier4_both_brands', 'tier4_gemini_only', 'tier4_claude_only']:
        tier_set = set(tier_categories[tier_name])
        tier_agreement = judge_disputed & tier_set
        tier_breakdown[tier_name] = len(tier_agreement)

        pct = len(tier_agreement) / len(tier_set) * 100 if len(tier_set) > 0 else 0
        print(f"{tier_name:25s}: {len(tier_agreement):3d}/{len(tier_set):3d} ({pct:5.1f}%)")

    print()

    # Analyze over-identification
    print("="*100)
    print("Over-Identification Analysis (Judge flagged but tiers say OK)")
    print("="*100)
    print()

    over_id_in_all_correct = over_id & all_correct
    over_id_in_tier5 = over_id & set(tier_categories['tier5_no_consensus'])
    over_id_in_failed = over_id & set(tier_categories['extraction_failed'])

    print(f"Judge flagged but models all agree with key:  {len(over_id_in_all_correct):3d}")
    print(f"Judge flagged but no model consensus (Tier 5): {len(over_id_in_tier5):3d}")
    print(f"Judge flagged but extraction failed:           {len(over_id_in_failed):3d}")
    print()

    if len(over_id_in_all_correct) > 20:
        print("⚠️  HIGH OVER-IDENTIFICATION: Judge disputed many questions where all 4 models agree with key")
        print("   This suggests potential judge bias or over-sensitivity")
    elif len(over_id_in_all_correct) > 10:
        print("⚠️  MODERATE OVER-IDENTIFICATION: Some questions disputed despite model agreement")
    else:
        print("✓  LOW OVER-IDENTIFICATION: Judge mostly agrees when models match key")
    print()

    # Family bias analysis
    print("="*100)
    print("Family Bias Analysis (When Gemini vs Claude Disagree)")
    print("="*100)
    print()

    bias_stats, bias_examples = analyze_family_bias(judge_data, all_answers)

    print(f"Of {bias_stats['total_disputed']} disputed questions:")
    print(f"  Judge matches Gemini ONLY:  {bias_stats['judge_matches_gemini_only']:3d} ({bias_stats['judge_matches_gemini_only']/bias_stats['total_disputed']*100:.1f}%)")
    print(f"  Judge matches Claude ONLY:  {bias_stats['judge_matches_claude_only']:3d} ({bias_stats['judge_matches_claude_only']/bias_stats['total_disputed']*100:.1f}%)")
    print(f"  Judge matches BOTH:         {bias_stats['judge_matches_both']:3d} ({bias_stats['judge_matches_both']/bias_stats['total_disputed']*100:.1f}%)")
    print(f"  Judge matches NEITHER:      {bias_stats['judge_matches_neither']:3d} ({bias_stats['judge_matches_neither']/bias_stats['total_disputed']*100:.1f}%)")
    print()

    # Calculate bias ratio
    bias_ratio = (bias_stats['judge_matches_gemini_only'] /
                  bias_stats['judge_matches_claude_only']) if bias_stats['judge_matches_claude_only'] > 0 else float('inf')

    print("="*100)
    print("Bias Ratio Analysis")
    print("="*100)
    print()
    print(f"Bias Ratio: {bias_ratio:.2f}x")
    print()

    if bias_ratio > 2.5:
        print("⚠️  STRONG FAMILY BIAS DETECTED")
        print(f"   Judge agrees with Gemini {bias_ratio:.1f}x more often than Claude")
        print("   when they disagree. This indicates significant self-agreement bias.")
    elif bias_ratio > 1.5:
        print("⚠️  MODERATE FAMILY BIAS")
        print(f"   Judge shows preference for Gemini answers ({bias_ratio:.1f}x)")
    elif bias_ratio < 0.67:
        print(f"   Unexpected: Judge favors Claude ({1/bias_ratio:.1f}x)")
    else:
        print("✓  NO SIGNIFICANT FAMILY BIAS")
        print("   Judge treats both model families fairly")
    print()

    # Show examples
    if bias_examples['gemini_only']:
        print("="*100)
        print("Examples: Judge Agrees with Gemini, Not Claude")
        print("="*100)
        print()
        for i, ex in enumerate(bias_examples['gemini_only'], 1):
            print(f"Example {i}: {ex['loc']}")
            print(f"  Answer Key:    {ex['answer_key']}")
            print(f"  Gemini Pro:    {ex['gemini_pro']}")
            print(f"  Gemini Flash:  {ex['gemini_flash']}")
            print(f"  Claude Opus:   {ex['claude_opus']}")
            print(f"  Claude Sonnet: {ex['claude_sonnet']}")
            print(f"  Judge Says:    {ex['judge']}")
            print(f"  Explanation:   {ex['explanation']}")
            print()

    # Recommendations
    print("="*100)
    print("Recommendations")
    print("="*100)
    print()

    print(f"1. Judge Reliability: Precision={precision:.1f}%, Recall={recall:.1f}%")
    if precision < 70:
        print("   → Judge has LOW precision - many flagged questions not validated by tiers")
        print("   → Recommend using tier-based validation as primary metric")
    elif precision >= 80:
        print("   → Judge has HIGH precision - most flagged questions are validated")
        print("   → Judge is reliable for identifying bad keys")

    print()
    print(f"2. Family Bias: Ratio={bias_ratio:.2f}x")
    if bias_ratio > 2.0:
        print("   → STRONG bias toward Gemini family detected")
        print("   → Do NOT use this judge for final validation")
        print("   → Use tier-based cross-model agreement instead")
    elif bias_ratio > 1.5:
        print("   → MODERATE bias detected")
        print("   → Use judge results with caution, cross-validate with tiers")
    else:
        print("   → NO significant bias detected")
        print("   → Judge can be trusted for validation")

    print()
    print(f"3. Best Validation Strategy:")
    if precision >= 80 and bias_ratio < 1.5:
        print("   → Use judge validation (reliable and unbiased)")
    elif len(agreement) >= 80:
        print("   → Use questions where BOTH judge AND tiers agree ({len(agreement)} questions)")
        print("   → This ensures highest confidence bad key identification")
    else:
        print("   → Use tier-based validation (Tier 1-2: {tier_breakdown['tier1_all_4_agree'] + tier_breakdown['tier2_3_agree']} questions)")
        print("   → More reliable than biased judge")

    print()
    print("="*100)

if __name__ == "__main__":
    main()
