#!/usr/bin/env python3
"""
Rigorous Bias Detection Analysis
Based on science-methodology-reviewer recommendations

Implements:
1. Analysis 1A: Gemini behavior in tier4_claude_only
2. Analysis 1B: Judge behavior conditional on Gemini agreement
3. Analysis 2A: Symmetric test on tier4_gemini_only
4. McNemar's test for paired disagreements
5. Permutation test for tier4 asymmetry
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import math

def load_all_data():
    """Load judge validation, model answers, and categorize questions"""

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
                'answer_key': entry.get('answer', '')
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

def analysis_1a_gemini_in_tier4_claude(judge_data, all_answers, tier4_claude_locs):
    """
    Analysis 1A: Examine Gemini behavior in tier4_claude_only questions

    Critical diagnostic: If Gemini mostly agrees with answer key, suggests Claude weakness.
    If Gemini shows scattered responses or agrees with Claude, suggests bad keys.
    """

    print("="*100)
    print("Analysis 1A: Gemini Behavior in tier4_claude_only (58 questions)")
    print("="*100)
    print()
    print("Key Diagnostic: What did Gemini models answer when only Claude pair disagreed?")
    print()

    results = {
        'gemini_pro_agrees_key': 0,
        'gemini_flash_agrees_key': 0,
        'both_gemini_agree_key': 0,
        'gemini_pro_agrees_claude': 0,
        'gemini_flash_agrees_claude': 0,
        'both_gemini_agree_claude': 0,
        'gemini_scattered': 0
    }

    details = []

    for loc in tier4_claude_locs:
        gemini_pro = all_answers['gemini-2-5-pro'][loc]['extracted']
        gemini_flash = all_answers['gemini-2-5-flash'][loc]['extracted']
        claude_opus = all_answers['claude-opus-4-1'][loc]['extracted']
        claude_sonnet = all_answers['claude-sonnet-4-5'][loc]['extracted']
        answer_key = all_answers['gemini-2-5-pro'][loc]['answer_key']

        claude_answer = claude_opus  # Both Claudes agree on this

        pro_agrees_key = (gemini_pro == answer_key)
        flash_agrees_key = (gemini_flash == answer_key)
        pro_agrees_claude = (gemini_pro == claude_answer)
        flash_agrees_claude = (gemini_flash == claude_answer)

        if pro_agrees_key:
            results['gemini_pro_agrees_key'] += 1
        if flash_agrees_key:
            results['gemini_flash_agrees_key'] += 1
        if pro_agrees_key and flash_agrees_key:
            results['both_gemini_agree_key'] += 1
        if pro_agrees_claude:
            results['gemini_pro_agrees_claude'] += 1
        if flash_agrees_claude:
            results['gemini_flash_agrees_claude'] += 1
        if pro_agrees_claude and flash_agrees_claude:
            results['both_gemini_agree_claude'] += 1
        if gemini_pro != answer_key and gemini_pro != claude_answer and \
           gemini_flash != answer_key and gemini_flash != claude_answer:
            results['gemini_scattered'] += 1

        details.append({
            'loc': loc,
            'answer_key': answer_key,
            'claude_answer': claude_answer,
            'gemini_pro': gemini_pro,
            'gemini_flash': gemini_flash,
            'pro_agrees_key': pro_agrees_key,
            'flash_agrees_key': flash_agrees_key,
            'pro_agrees_claude': pro_agrees_claude,
            'flash_agrees_claude': flash_agrees_claude
        })

    total = len(tier4_claude_locs)

    print(f"Gemini Pro agrees with answer key:    {results['gemini_pro_agrees_key']:3d} / {total} ({results['gemini_pro_agrees_key']/total*100:.1f}%)")
    print(f"Gemini Flash agrees with answer key:  {results['gemini_flash_agrees_key']:3d} / {total} ({results['gemini_flash_agrees_key']/total*100:.1f}%)")
    print(f"BOTH Gemini agree with answer key:    {results['both_gemini_agree_key']:3d} / {total} ({results['both_gemini_agree_key']/total*100:.1f}%)")
    print()
    print(f"Gemini Pro agrees with Claude:        {results['gemini_pro_agrees_claude']:3d} / {total} ({results['gemini_pro_agrees_claude']/total*100:.1f}%)")
    print(f"Gemini Flash agrees with Claude:      {results['gemini_flash_agrees_claude']:3d} / {total} ({results['gemini_flash_agrees_claude']/total*100:.1f}%)")
    print(f"BOTH Gemini agree with Claude:        {results['both_gemini_agree_claude']:3d} / {total} ({results['both_gemini_agree_claude']/total*100:.1f}%)")
    print()
    print(f"Gemini models scattered (neither key nor Claude): {results['gemini_scattered']:3d} / {total} ({results['gemini_scattered']/total*100:.1f}%)")
    print()

    # Aggregate Gemini agreement rate with key
    gemini_total_responses = total * 2  # Two Gemini models
    gemini_key_agreements = results['gemini_pro_agrees_key'] + results['gemini_flash_agrees_key']
    gemini_key_agreement_rate = gemini_key_agreements / gemini_total_responses * 100

    print("="*100)
    print("Diagnostic Interpretation")
    print("="*100)
    print()
    print(f"Gemini Key Agreement Rate: {gemini_key_agreement_rate:.1f}%")
    print()

    if gemini_key_agreement_rate > 70:
        print("✓ HIGH agreement with answer key (>70%)")
        print("  → Interpretation: Claude models are genuinely wrong in tier4_claude_only")
        print("  → Answer keys are likely correct")
        print("  → Bias evidence: WEAK")
    elif gemini_key_agreement_rate < 50:
        print("✓ LOW agreement with answer key (<50%)")
        print("  → Interpretation: Gemini also struggles with these questions")
        print("  → Answer keys are likely incorrect")
        print("  → Bias evidence: STRONG")
    else:
        print("⚠ MODERATE agreement with answer key (50-70%)")
        print("  → Interpretation: Mixed - some bad keys, some genuine Claude errors")
        print("  → Bias evidence: MODERATE")

    print()

    # Check Claude agreement
    gemini_claude_agreements = results['gemini_pro_agrees_claude'] + results['gemini_flash_agrees_claude']
    gemini_claude_agreement_rate = gemini_claude_agreements / gemini_total_responses * 100

    print(f"Gemini-Claude Agreement Rate: {gemini_claude_agreement_rate:.1f}%")
    print()

    if gemini_claude_agreement_rate > 40:
        print("✓ SUBSTANTIAL Gemini-Claude agreement (>40%)")
        print("  → Interpretation: Cross-family consensus that answer keys are wrong")
        print("  → Bias evidence: STRONG")
    else:
        print("  → LOW Gemini-Claude agreement")
        print("  → Models genuinely disagree on these questions")

    print()

    return results, details

def analysis_1b_judge_conditional(judge_data, tier4_claude_details):
    """
    Analysis 1B: Judge behavior conditional on Gemini agreement patterns

    Tests if judge is more likely to invalidate when Gemini agrees with Claude
    vs when Gemini agrees with answer key.
    """

    print("="*100)
    print("Analysis 1B: Judge Behavior Conditional on Gemini Agreement")
    print("="*100)
    print()

    subsets = {
        'both_gemini_agree_key': [],
        'one_gemini_agrees_key': [],
        'both_gemini_agree_claude': [],
        'one_gemini_agrees_claude': [],
        'gemini_scattered': []
    }

    for detail in tier4_claude_details:
        loc = detail['loc']

        pro_key = detail['pro_agrees_key']
        flash_key = detail['flash_agrees_key']
        pro_claude = detail['pro_agrees_claude']
        flash_claude = detail['flash_agrees_claude']

        if pro_key and flash_key:
            subsets['both_gemini_agree_key'].append(loc)
        elif pro_key or flash_key:
            subsets['one_gemini_agrees_key'].append(loc)
        elif pro_claude and flash_claude:
            subsets['both_gemini_agree_claude'].append(loc)
        elif pro_claude or flash_claude:
            subsets['one_gemini_agrees_claude'].append(loc)
        else:
            subsets['gemini_scattered'].append(loc)

    print("Subsets based on Gemini behavior:")
    print()

    for subset_name, locs in subsets.items():
        if not locs:
            continue

        invalidated = sum(1 for loc in locs if judge_data[loc]['answer_key_correct'] == False)
        total = len(locs)
        rate = invalidated / total * 100 if total > 0 else 0

        print(f"{subset_name:30s}: {invalidated:2d}/{total:2d} ({rate:5.1f}%) invalidated by judge")

    print()
    print("="*100)
    print("Interpretation")
    print("="*100)
    print()

    both_key = len(subsets['both_gemini_agree_key'])
    both_key_inv = sum(1 for loc in subsets['both_gemini_agree_key'] if judge_data[loc]['answer_key_correct'] == False)
    both_key_rate = both_key_inv / both_key * 100 if both_key > 0 else 0

    both_claude = len(subsets['both_gemini_agree_claude'])
    both_claude_inv = sum(1 for loc in subsets['both_gemini_agree_claude'] if judge_data[loc]['answer_key_correct'] == False)
    both_claude_rate = both_claude_inv / both_claude * 100 if both_claude > 0 else 0

    print(f"When BOTH Gemini agree with key:   {both_key_rate:.1f}% invalidation")
    print(f"When BOTH Gemini agree with Claude: {both_claude_rate:.1f}% invalidation")
    print()

    if both_claude_rate > both_key_rate * 2:
        print("✓ Judge invalidates MORE when Gemini agrees with Claude")
        print("  → Consistent with unbiased judge (cross-family agreement = stronger evidence)")
    elif both_key_rate > both_claude_rate:
        print("⚠ Judge invalidates MORE when Gemini agrees with key")
        print("  → Inconsistent pattern - possible bias")
    else:
        print("  → Similar invalidation rates across patterns")

    print()

def analysis_2a_symmetric_test(judge_data, all_answers, tier4_gemini_locs):
    """
    Analysis 2A: Symmetric test on tier4_gemini_only

    Mirrors Analysis 1A - examines Claude behavior when only Gemini pair disagrees.
    """

    print("="*100)
    print("Analysis 2A: Symmetric Test - Claude Behavior in tier4_gemini_only (30 questions)")
    print("="*100)
    print()
    print("Key Diagnostic: What did Claude models answer when only Gemini pair disagreed?")
    print()

    results = {
        'claude_opus_agrees_key': 0,
        'claude_sonnet_agrees_key': 0,
        'both_claude_agree_key': 0,
        'claude_opus_agrees_gemini': 0,
        'claude_sonnet_agrees_gemini': 0,
        'both_claude_agree_gemini': 0,
        'claude_scattered': 0
    }

    for loc in tier4_gemini_locs:
        gemini_pro = all_answers['gemini-2-5-pro'][loc]['extracted']
        gemini_flash = all_answers['gemini-2-5-flash'][loc]['extracted']
        claude_opus = all_answers['claude-opus-4-1'][loc]['extracted']
        claude_sonnet = all_answers['claude-sonnet-4-5'][loc]['extracted']
        answer_key = all_answers['gemini-2-5-pro'][loc]['answer_key']

        gemini_answer = gemini_pro  # Both Gemini agree on this

        opus_agrees_key = (claude_opus == answer_key)
        sonnet_agrees_key = (claude_sonnet == answer_key)
        opus_agrees_gemini = (claude_opus == gemini_answer)
        sonnet_agrees_gemini = (claude_sonnet == gemini_answer)

        if opus_agrees_key:
            results['claude_opus_agrees_key'] += 1
        if sonnet_agrees_key:
            results['claude_sonnet_agrees_key'] += 1
        if opus_agrees_key and sonnet_agrees_key:
            results['both_claude_agree_key'] += 1
        if opus_agrees_gemini:
            results['claude_opus_agrees_gemini'] += 1
        if sonnet_agrees_gemini:
            results['claude_sonnet_agrees_gemini'] += 1
        if opus_agrees_gemini and sonnet_agrees_gemini:
            results['both_claude_agree_gemini'] += 1
        if claude_opus != answer_key and claude_opus != gemini_answer and \
           claude_sonnet != answer_key and claude_sonnet != gemini_answer:
            results['claude_scattered'] += 1

    total = len(tier4_gemini_locs)

    print(f"Claude Opus agrees with answer key:   {results['claude_opus_agrees_key']:3d} / {total} ({results['claude_opus_agrees_key']/total*100:.1f}%)")
    print(f"Claude Sonnet agrees with answer key: {results['claude_sonnet_agrees_key']:3d} / {total} ({results['claude_sonnet_agrees_key']/total*100:.1f}%)")
    print(f"BOTH Claude agree with answer key:    {results['both_claude_agree_key']:3d} / {total} ({results['both_claude_agree_key']/total*100:.1f}%)")
    print()
    print(f"Claude Opus agrees with Gemini:       {results['claude_opus_agrees_gemini']:3d} / {total} ({results['claude_opus_agrees_gemini']/total*100:.1f}%)")
    print(f"Claude Sonnet agrees with Gemini:     {results['claude_sonnet_agrees_gemini']:3d} / {total} ({results['claude_sonnet_agrees_gemini']/total*100:.1f}%)")
    print(f"BOTH Claude agree with Gemini:        {results['both_claude_agree_gemini']:3d} / {total} ({results['both_claude_agree_gemini']/total*100:.1f}%)")
    print()
    print(f"Claude models scattered (neither key nor Gemini): {results['claude_scattered']:3d} / {total} ({results['claude_scattered']/total*100:.1f}%)")
    print()

    claude_total_responses = total * 2
    claude_key_agreements = results['claude_opus_agrees_key'] + results['claude_sonnet_agrees_key']
    claude_key_agreement_rate = claude_key_agreements / claude_total_responses * 100

    print(f"Claude Key Agreement Rate: {claude_key_agreement_rate:.1f}%")
    print()

    # Judge invalidation rate
    judge_invalidated = sum(1 for loc in tier4_gemini_locs if judge_data[loc]['answer_key_correct'] == False)
    judge_invalidation_rate = judge_invalidated / total * 100

    print(f"Judge Invalidation Rate: {judge_invalidation_rate:.1f}%")
    print()

    return claude_key_agreement_rate, judge_invalidation_rate

def mcnemar_test(judge_data, all_answers):
    """
    McNemar's test for paired disagreements

    Tests if the 43 vs 2 asymmetry (judge agrees with Gemini vs Claude when they disagree)
    is statistically significant.
    """

    print("="*100)
    print("McNemar's Test: Judge Family Preference When Models Disagree")
    print("="*100)
    print()

    judge_matches_gemini_only = 0
    judge_matches_claude_only = 0

    for loc, judge_info in judge_data.items():
        if judge_info['answer_key_correct'] != False:
            continue

        gemini_pro = all_answers['gemini-2-5-pro'][loc]['extracted']
        gemini_flash = all_answers['gemini-2-5-flash'][loc]['extracted']
        claude_opus = all_answers['claude-opus-4-1'][loc]['extracted']
        claude_sonnet = all_answers['claude-sonnet-4-5'][loc]['extracted']

        judge_ans = judge_info['judge_answer']

        matches_gemini = judge_ans in [gemini_pro, gemini_flash]
        matches_claude = judge_ans in [claude_opus, claude_sonnet]

        if matches_gemini and not matches_claude:
            judge_matches_gemini_only += 1
        elif matches_claude and not matches_gemini:
            judge_matches_claude_only += 1

    print(f"Judge matches Gemini ONLY (not Claude): {judge_matches_gemini_only}")
    print(f"Judge matches Claude ONLY (not Gemini): {judge_matches_claude_only}")
    print()

    # McNemar's test
    b = judge_matches_gemini_only
    c = judge_matches_claude_only

    if b + c > 0:
        chi2 = (b - c) ** 2 / (b + c)

        # Approximate p-value for chi-square with df=1
        # Using standard chi-square critical values
        if chi2 > 10.828:
            p_value = 0.001  # p < 0.001
        elif chi2 > 6.635:
            p_value = 0.01   # p < 0.01
        elif chi2 > 3.841:
            p_value = 0.05   # p < 0.05
        else:
            # More precise approximation using error function
            z = math.sqrt(chi2)
            # Standard normal approximation
            p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

        print(f"McNemar's χ² = {chi2:.2f}")
        print(f"p-value = {p_value:.2e}")
        print()

        if p_value < 0.001:
            print("*** EXTREMELY SIGNIFICANT (p < 0.001) ***")
            print("The judge's preference for Gemini is NOT due to chance.")
            print("Strong evidence of family bias.")
        elif p_value < 0.01:
            print("** HIGHLY SIGNIFICANT (p < 0.01) **")
            print("Strong evidence of family bias.")
        elif p_value < 0.05:
            print("* SIGNIFICANT (p < 0.05) *")
            print("Evidence of family bias.")
        else:
            print("NOT SIGNIFICANT (p >= 0.05)")
            print("Asymmetry could be due to chance.")

    print()

    return chi2, p_value

def permutation_test(judge_data, tier4_gemini_locs, tier4_claude_locs, n_permutations=10000):
    """
    Permutation test for tier4 invalidation asymmetry

    Tests if the 46.7% vs 8.6% difference is statistically significant.
    """

    print("="*100)
    print("Permutation Test: Tier 4 Invalidation Asymmetry")
    print("="*100)
    print()

    # Observed rates
    gemini_invalidated = sum(1 for loc in tier4_gemini_locs if judge_data[loc]['answer_key_correct'] == False)
    claude_invalidated = sum(1 for loc in tier4_claude_locs if judge_data[loc]['answer_key_correct'] == False)

    gemini_rate = gemini_invalidated / len(tier4_gemini_locs) * 100
    claude_rate = claude_invalidated / len(tier4_claude_locs) * 100
    observed_diff = gemini_rate - claude_rate

    print(f"Observed:")
    print(f"  tier4_gemini_only: {gemini_invalidated}/{len(tier4_gemini_locs)} = {gemini_rate:.1f}% invalidation")
    print(f"  tier4_claude_only: {claude_invalidated}/{len(tier4_claude_locs)} = {claude_rate:.1f}% invalidation")
    print(f"  Difference: {observed_diff:.1f} percentage points")
    print()

    # Permutation test
    all_locs = tier4_gemini_locs + tier4_claude_locs
    all_invalidations = [judge_data[loc]['answer_key_correct'] == False for loc in all_locs]

    n_gemini = len(tier4_gemini_locs)

    diffs = []
    for _ in range(n_permutations):
        # Randomly shuffle labels
        np.random.shuffle(all_invalidations)

        perm_gemini = all_invalidations[:n_gemini]
        perm_claude = all_invalidations[n_gemini:]

        perm_gemini_rate = sum(perm_gemini) / len(perm_gemini) * 100
        perm_claude_rate = sum(perm_claude) / len(perm_claude) * 100
        perm_diff = perm_gemini_rate - perm_claude_rate

        diffs.append(perm_diff)

    # p-value
    p_value = sum(1 for d in diffs if abs(d) >= abs(observed_diff)) / n_permutations

    print(f"Permutation test ({n_permutations} permutations):")
    print(f"p-value = {p_value:.4f}")
    print()

    if p_value < 0.001:
        print("*** EXTREMELY SIGNIFICANT (p < 0.001) ***")
        print("The asymmetry is NOT due to chance.")
    elif p_value < 0.01:
        print("** HIGHLY SIGNIFICANT (p < 0.01) **")
    elif p_value < 0.05:
        print("* SIGNIFICANT (p < 0.05) *")
    else:
        print("NOT SIGNIFICANT (p >= 0.05)")

    print()

    return observed_diff, p_value

def main():
    print("="*100)
    print("RIGOROUS BIAS DETECTION ANALYSIS")
    print("Based on Science Methodology Review")
    print("="*100)
    print()

    # Load data
    print("Loading data...")
    judge_data, all_answers = load_all_data()
    print("✓ Data loaded")
    print()

    # Categorize questions
    print("Categorizing questions into tiers...")
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

    print(f"✓ Categorization complete")
    print(f"  tier4_claude_only: {len(tier_categories['tier4_claude_only'])} questions")
    print(f"  tier4_gemini_only: {len(tier_categories['tier4_gemini_only'])} questions")
    print()

    # Run analyses
    tier4_claude_locs = tier_categories['tier4_claude_only']
    tier4_gemini_locs = tier_categories['tier4_gemini_only']

    # Analysis 1A
    results_1a, details_1a = analysis_1a_gemini_in_tier4_claude(
        judge_data, all_answers, tier4_claude_locs
    )

    # Analysis 1B
    analysis_1b_judge_conditional(judge_data, details_1a)

    # Analysis 2A
    claude_key_rate, judge_inv_rate = analysis_2a_symmetric_test(
        judge_data, all_answers, tier4_gemini_locs
    )

    # McNemar's test
    chi2, p_mcnemar = mcnemar_test(judge_data, all_answers)

    # Permutation test
    obs_diff, p_perm = permutation_test(judge_data, tier4_gemini_locs, tier4_claude_locs)

    # Final Summary
    print("="*100)
    print("FINAL SUMMARY: BIAS EVIDENCE STRENGTH")
    print("="*100)
    print()

    # Calculate key metrics from Analysis 1A
    gemini_total = len(tier4_claude_locs) * 2
    gemini_key_agreements = results_1a['gemini_pro_agrees_key'] + results_1a['gemini_flash_agrees_key']
    gemini_key_rate = gemini_key_agreements / gemini_total * 100

    evidence_score = 0

    print("Evidence Components:")
    print()

    # 1. Gemini behavior in tier4_claude_only
    print(f"1. Gemini Key Agreement in tier4_claude_only: {gemini_key_rate:.1f}%")
    if gemini_key_rate < 50:
        print("   ✓ LOW (<50%) → Strong bias evidence (+3 points)")
        evidence_score += 3
    elif gemini_key_rate < 60:
        print("   ⚠ MODERATE (50-60%) → Moderate bias evidence (+2 points)")
        evidence_score += 2
    elif gemini_key_rate < 70:
        print("   ⚠ MODERATE-HIGH (60-70%) → Weak bias evidence (+1 point)")
        evidence_score += 1
    else:
        print("   ✗ HIGH (>70%) → Suggests Claude weakness (0 points)")
    print()

    # 2. McNemar's test
    print(f"2. McNemar's Test: p = {p_mcnemar:.2e}")
    if p_mcnemar < 0.001:
        print("   ✓ EXTREMELY SIGNIFICANT → Strong bias evidence (+3 points)")
        evidence_score += 3
    elif p_mcnemar < 0.01:
        print("   ✓ HIGHLY SIGNIFICANT → Moderate bias evidence (+2 points)")
        evidence_score += 2
    elif p_mcnemar < 0.05:
        print("   ⚠ SIGNIFICANT → Weak bias evidence (+1 point)")
        evidence_score += 1
    else:
        print("   ✗ NOT SIGNIFICANT (0 points)")
    print()

    # 3. Permutation test
    print(f"3. Permutation Test: p = {p_perm:.4f}")
    if p_perm < 0.001:
        print("   ✓ EXTREMELY SIGNIFICANT → Strong bias evidence (+3 points)")
        evidence_score += 3
    elif p_perm < 0.01:
        print("   ✓ HIGHLY SIGNIFICANT → Moderate bias evidence (+2 points)")
        evidence_score += 2
    elif p_perm < 0.05:
        print("   ⚠ SIGNIFICANT → Weak bias evidence (+1 point)")
        evidence_score += 1
    else:
        print("   ✗ NOT SIGNIFICANT (0 points)")
    print()

    # 4. Symmetric test
    print(f"4. Symmetric Test - Claude Key Agreement in tier4_gemini_only: {claude_key_rate:.1f}%")
    if abs(claude_key_rate - gemini_key_rate) > 20:
        print(f"   ✓ LARGE ASYMMETRY (>{abs(claude_key_rate - gemini_key_rate):.0f}pp difference) → Bias evidence (+2 points)")
        evidence_score += 2
    elif abs(claude_key_rate - gemini_key_rate) > 10:
        print(f"   ⚠ MODERATE ASYMMETRY (>{abs(claude_key_rate - gemini_key_rate):.0f}pp difference) → Weak bias evidence (+1 point)")
        evidence_score += 1
    else:
        print(f"   ✗ SYMMETRIC (0 points)")
    print()

    print("="*100)
    print(f"TOTAL EVIDENCE SCORE: {evidence_score}/11")
    print()

    if evidence_score >= 9:
        print("*** VERDICT: STRONG BIAS DETECTED ***")
        print("Recommendation: Do NOT use Gemini as judge for validation")
        print("Use tier-based cross-model agreement instead")
    elif evidence_score >= 6:
        print("** VERDICT: MODERATE BIAS DETECTED **")
        print("Recommendation: Use judge results with caution")
        print("Cross-validate with tier-based analysis")
    elif evidence_score >= 3:
        print("* VERDICT: WEAK BIAS DETECTED *")
        print("Recommendation: Some evidence of bias, but not conclusive")
    else:
        print("VERDICT: NO STRONG BIAS EVIDENCE")
        print("Judge appears relatively unbiased")

    print()
    print("="*100)

if __name__ == "__main__":
    main()
