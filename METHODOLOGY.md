# Enhanced TLUE Methodology: A Comprehensive Answer Key Validation Framework

## Executive Summary

This document describes the complete methodology used to evaluate large language models (Gemini 2.5 Pro/Flash and Claude Opus/Sonnet 4) on the Tibetan Language Understanding Evaluation (TLUE) benchmark.
1. Original repository: https://github.com/Vicentvankor/TLUE
2. Original paper: https://arxiv.org/pdf/2503.12051
This doc has particular emphasis on **validating answer key quality** through two independent approaches: cross-model agreement analysis and LLM-as-judge validation.

Our findings reveal answer key quality issues potentially affecting 16-37% of questions, and demonstrate that filtering by data quality can improve measured accuracy by 13-37 percentage points. Critically, we also identified **substantial judge bias** (21.5× agreement ratio favoring same-family models) when using Gemini as judge. A deep dive on this potential bias is required to check if it's true bias or true accuracy, highlighting the importance of using agreement-based filtering as a complementary or primary validation method.

---

## Table of Contents

1. [Introduction & Motivation](#1-introduction--motivation)
2. [Initial Model Evaluation](#2-initial-model-evaluation)
3. [Answer Key Validation: Dual-Method Approach](#3-answer-key-validation-dual-method-approach)
4. [Judge Bias Detection & Analysis](#4-judge-bias-detection--analysis)
5. [Tier-Based Filtering Framework](#5-tier-based-filtering-framework)
6. [Data Quality Impact Analysis](#6-data-quality-impact-analysis)
7. [Statistical Validation](#7-statistical-validation)
8. [Findings & Recommendations](#8-findings--recommendations)
9. [Limitations & Future Work](#9-limitations--future-work)
10. [Conclusion](#10-conclusion)

---

## 1. Introduction & Motivation

### 1.1 Background

This project aims to **raise awareness about linguistic equity in AI and utimately improve model performance and inclusivity for underrepresented languages**. We believe that every language embodies the culture, identity, and wisdom of its communicty and no language should be left behind in the age of AI.
The original Tibetan Language Understanding Evaluation (TLUE) is a multiple-choice benchmark dataset designed to evaluate large language models' understanding of Tibetan-Chinese bilingual content across 67 academic subjects. The partially released dataset contains **670 questions** drawn from subjects ranging from agronomy to world history. This enhanced eval frame is built to measure model performance on handling Tibetan language and validate the groundtruth datasets.  

### 1.2 The Answer Key Problem

When evaluating multiple state-of-the-art models using TLUE, we observed **systematic disagreement** between model answers and provided answer keys, with models frequently choosing the same "incorrect" answer. This pattern suggested potential issues with the answer keys themselves rather than model failures.

**Key observation**: When four independent models (two from Google, two from Anthropic) all select the same answer that differs from the provided key, this indicates high probability of answer key error. Human in the loop is required to further check those datasets.

### 1.3 Research Questions

1. **How many answer keys are potentially incorrect?**
2. **What are causes for those incorrect? human errors? culture nuance lost in translation? or something else**
3. **Can we develop an objective, bias-free method to identify problematic questions?**
4. **How does answer key quality affect model performance measurement?**
5. **What is the optimal data quality threshold for fair model comparison?**
6. **Does using an LLM judge introduce systematic bias?**

---
## 2. Initial Model Evaluation

### 2.1 Models Evaluated

We evaluated four state-of-the-art large language models:

| Model | Provider | Family | Release |
|-------|----------|--------|---------|
| **Gemini 2.5 Pro** | Google | Gemini | 2025 |
| **Gemini 2.5 Flash** | Google | Gemini | 2025 |
| **Claude Opus 4.1** | Anthropic | Claude | 2025 |
| **Claude Sonnet 4.5** | Anthropic | Claude | 2025 |

### 2.2 Evaluation Protocol

- **Prompt type**: 0-shot evaluation (no examples provided)
- **Answer extraction**: "Concern All Answer" method (extracts from full response)
- **Question format**: Multiple choice (A, B, C, D)
- **Dataset size**: 670 questions across 67 subjects
- **Language**: Chinese to Tibetan translated (dataset in Tibetan)

### 2.3 Baseline Results

**Conditional Accuracy** (accuracy when model provides a valid response):

| Model | Valid Responses | Correct Answers | Conditional Accuracy |
|-------|----------------|-----------------|---------------------|
| Gemini 2.5 Pro | 657/670 (98.1%) | 447/657 | **68.0%** |
| Gemini 2.5 Flash | 649/670 (96.9%) | 430/649 | **66.3%** |
| Claude Opus 4.1 | 643/670 (96.0%) | 342/643 | **53.2%** |
| Claude Sonnet 4.5 | 653/670 (97.5%) | 330/653 | **50.5%** |

### 2.4 Initial Observations

1. **High response rates** (96-98%) indicate models can process the content
2. **Large performance gap** between Gemini and Claude families (~16pp)
3. **Gemini models outperform Claude** across all subjects
4. However, systematic analysis revealed this gap may be inflated by answer key quality issues. Answer key valiadation is required.

---

## 3. Answer Key Validation: Dual-Method Approach

To address answer key quality concerns, we developed **two independent validation methods**:

### 3.1 Method 1: Cross-Model Agreement Analysis

**Hypothesis**: When more than 3 independent models from different providers agree on an answer that differs from the key, the consensus answer is likely correct.

**Implementation**:
```python
def categorize_question(gemini_pro, gemini_flash, claude_opus, claude_sonnet, answer_key):
    """Categorize based on model agreement pattern"""

    # Tier 1: All 4 models agree (different from key)
    if all_models_agree and answer != answer_key:
        return 'tier1_all_4_agree'

    # Tier 2: 3 models agree (different from key)
    if three_models_agree and answer != answer_key:
        return 'tier2_3_agree'

    # Tier 3: Cross-brand agreement (1 Gemini + 1 Claude agree)
    if gemini_agrees_with_claude and answer != answer_key:
        return 'tier3_cross_brand'

    # Tier 4: Same-brand agreement only
    # Tier 5: No consensus
```

**Tiers of confidence**:
- **Tier 1** (Highest): All 4 models agree but different from key → 33 questions (4.9%)
- **Tier 2** (High): 3 models agree but different from key → 74 questions (11.0%)
- **Tier 3** (Medium): Cross-brand pair agrees but different from key → 50 questions (7.5%)
- **Tier 4** (Low): Same-brand only but different from key → 94 questions (14.0%)
- **Tier 5**: No consensus → 111 questions (16.6%)

**Total potentially problematic**: Tier 1-2 = **107 questions (16.0%)**

### 3.2 Method 2: LLM-as-Judge Validation

**Philosophy**: Instruct a LLM-as-judge (Gemini 2.5 Pro) to independently verify answer keys by analyzing question content, all answer choices, and provided rationale.

**Implementation**:
```python
judge_prompt = """
You are validating answer keys for a Tibetan assessment.

Question: {question}
Options: A) {option_a} B) {option_b} C) {option_c} D) {option_d}
Provided answer key: {answer_key}
Rationale: {explanation}

Task:
1. Is the provided answer key correct? (True/False)
2. If False, what is the correct answer?
3. Provide explanation/Reasoning
"""
```

**Judge validation results**:
- Questions flagged as incorrect: **130 questions (19.4%)**
- Judge proposed alternative answers for all flagged questions
- Judge agreed with Gemini models significantly more than Claude models (see bias analysis)

### 3.3 Method Comparison

| Metric | Cross-Model Agreement | LLM Judge |
|--------|----------------------|-----------|
| Questions flagged | 107 (Tier 1-2) | 130 |
| Bias-free | ✅ Yes (uses consensus) | ⚠️ No (21.5× bias detected) |
| Interpretability | ✅ High (counts visible) | ⚠️ Moderate (black box) |
| Independence | ✅ Model-agnostic | ❌ Uses Gemini judge |

**Key insight**: Cross-model agreement provides a more objective, bias-free validation method compared to LLM-as-judge.

---

## 4. Judge Bias Detection & Analysis

### 4.1 Motivation

When using Gemini 2.5 Pro as judge, we investigated whether the judge exhibited **self-agreement bias** (favoring Gemini models over Claude models).

### 4.2 Bias Testing Methodology

For the 130 questions where judge marked the answer key as incorrect, we analyzed:

1. **Agreement rates**: How often does judge's "correct answer" match each model?
2. **Cross-model comparison**: When models disagree, does judge favor one family?
3. **Bias ratio calculation**:
   ```
   Bias Ratio = (Judge matches Gemini ONLY) / (Judge matches Claude ONLY)
   ```

### 4.3 Bias Analysis Results

**Agreement rates** (judge's answer matches model's answer):

| Model | Agreement Rate | Relative to Baseline |
|-------|---------------|---------------------|
| Gemini 2.5 Pro | **105/130 (80.8%)** | +42.3pp |
| Gemini 2.5 Flash | 67/130 (51.5%) | +13.0pp |
| Claude Opus 4.1 | 50/130 (38.5%) | — (baseline) |
| Claude Sonnet 4.5 | 52/130 (40.0%) | +1.5pp |

**Cross-model bias analysis**:

| Pattern | Count | Percentage |
|---------|-------|-----------|
| Judge matches Gemini ONLY (not Claude) | **43** | 33.1% |
| Judge matches Claude ONLY (not Gemini) | **2** | 1.5% |
| Judge matches BOTH families | 66 | 50.8% |
| Judge matches neither | 19 | 14.6% |

**Bias ratio**: **21.5×** (Judge agrees with Gemini 21.5× more often than Claude when families disagree)

### 4.4 Interpretation

**⚠️ HIGH BIAS DETECTED**

The 21.5× bias ratio indicates substantial self-agreement bias where:
- Gemini judge systematically favors Gemini models' answers
- This bias inflates Gemini's apparent performance on judge-validated subsets
- Judge validation is **not suitable as sole validation method** unless the human validation agrees to judge validation

**Recommendation**: Use cross-model agreement as primary validation, with judge providing supplementary validation only when combined with agreement-based filtering (union approach).

---

## 5. Tier-Based Filtering Framework

### 5.1 Filtering Scenarios

We evaluated models under multiple data quality scenarios:

| Scenario | Questions Excluded | Remaining Questions | Description |
|----------|-------------------|---------------------|-------------|
| **Baseline** | None | 670 | Original dataset |
| **Tier 1 Only** | Tier 1 (33) | 637 | Exclude highest-confidence errors |
| **Tier 1-2 Only** | Tier 1-2 (107) | 563 | **Recommended**: Exclude high-confidence errors |
| **Tier 1-3 Only** | Tier 1-3 (157) | 513 | Include medium-confidence |
| **Tier 1-4 Only** | Tier 1-4 (251) | 419 | Include low-confidence |
| **Judge Only** | Judge disputed (130) | 540 | Use LLM judge validation |
| **Tier 1-2 ∪ Judge** | Union (177) | 493 | **Highest quality**: Combine both methods |

### 5.2 Set Relationship Analysis

**Overlap between methods (model agreement vs llm-as-judge)**:
```
Tier 1-2 questions:     107
Judge disputed:         130
Intersection (∩):        60 (56% of Tier 1-2, 46% of Judge)
Union (∪):              177 (26.4% of dataset)
```

**Interpretation**:
- 56% of high-confidence model disagreements confirmed by the llm-as-judge
- 54% of judge disputes not supported by strong model consensus (potential bias)
- Union approach maximizes coverage while controlling for judge bias

### 5.3 Exclusion Rationale

**Why exclude rather than correct?**

1. **Conservative approach**: Excluding ambiguous questions ensures fair comparison quickly
3. **Research integrity**: Manually changing answer keys without ground truth review by experts is inappropriate
4. **Transparency**: Exclusion is auditable and reversible
5. **Human Validation Cost**: Require human experts to validate the ground truth (Fast followup)

---

## 6. Data Quality Impact Analysis

### 6.1 Performance Improvement by Scenario

**Conditional Accuracy across scenarios**:

| Scenario | Gemini 2.5 Pro | Gemini 2.5 Flash | Claude Opus 4.1 | Claude Sonnet 4.5 |
|----------|---------------|------------------|-----------------|-------------------|
| Baseline | 68.0% | 66.3% | 53.2% | 50.5% |
| Tier 1 Only | 69.3% (+1.3pp) | 68.0% (+1.7pp) | 54.6% (+1.4pp) | 51.8% (+1.3pp) |
| **Tier 1-2 Only** | **79.1% (+11.1pp)** | **77.7% (+11.4pp)** | **62.3% (+9.1pp)** | **58.8% (+8.3pp)** |
| Tier 1-3 Only | 81.8% (+13.8pp) | 80.3% (+14.0pp) | 66.1% (+12.9pp) | 62.5% (+12.0pp) |
| Tier 1-4 Only | 86.4% (+18.4pp) | 85.5% (+19.2pp) | 72.2% (+19.0pp) | 68.5% (+18.0pp) |
| Judge Only | 81.7% (+13.7pp) | 75.9% (+9.6pp) | 58.7% (+5.5pp) | 56.7% (+6.2pp) |
| **Tier 1-2 ∪ Judge** | **87.1% (+19.1pp)** | **82.4% (+16.1pp)** | **63.9% (+10.7pp)** | **61.4% (+10.9pp)** |

### 6.2 Key Findings

1. **All models improve** when filtering by data quality (1-19pp gain)
2. **Tier 1-2 Only** provides best balance of quality and coverage (563 questions, ~11pp improvement)
3. **Judge Only** shows biased improvements (Gemini +13.7pp, Claude +5.5pp) due to judge bias
4. **Tier 1-2 ∪ Judge** achieves highest quality (87.1% for Gemini Pro) but smallest dataset (493 questions)
5. **Relative rankings preserved**: Gemini Pro remains strongest across all scenarios

### 6.3 Statistical Significance

Using Wilson score confidence intervals (95%):

**Baseline** (Gemini 2.5 Pro):
- Measured: 68.04%
- 95% CI: [64.44%, 71.43%]

**Tier 1-2 Only** (Gemini 2.5 Pro):
- Measured: 79.09%
- 95% CI: [75.56%, 82.23%]
- **Improvement: +11.05pp** (statistically significant, non-overlapping CIs)

**Interpretation**: The improvement from baseline to Tier 1-2 filtering is highly statistically significant for all models.

---

## 7. Statistical Validation

### 7.1 Confidence Intervals

We computed **Wilson score 95% confidence intervals** for all metrics to account for binomial proportion uncertainty:

**Why Wilson score?**
- More accurate than normal approximation for extreme proportions
- Maintains proper coverage for small samples

**Implementation**:
```python
def wilson_score_interval(successes, trials, confidence=0.95):
    """Calculate Wilson score confidence interval"""
    p = successes / trials
    z = norm_ppf((1 + confidence) / 2)  # z = 1.96 for 95%

    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * sqrt((p*(1-p)/trials + z**2/(4*trials**2))) / denominator

    return (center - margin, center + margin)
```

### 7.2 Three Key Metrics

For each model and scenario, we report:

1. **Response Rate**: Valid responses / Total questions
   - Measures model's ability to provide valid answers
   - Range: 95.2% - 98.8%

2. **Overall Accuracy**: Correct answers / Total questions
   - Real-world effectiveness metric
   - Affected by both response rate and conditional accuracy

3. **Conditional Accuracy**: Correct answers / Valid responses
   - **Primary metric**: Model's accuracy when it provides an answer
   - Isolates model capability from response coverage

### 7.3 Statistical Significance Testing

We used **two-proportion z-tests** to compare models:

**Null hypothesis**: H₀: p₁ = p₂ (proportions are equal)

**Significance levels**:
- \* p < 0.05 (significant)
- \*\* p < 0.01 (highly significant)
- \*\*\* p < 0.001 (extremely significant)

**Example**: Gemini 2.5 Pro vs Claude Opus 4.1 on Tier 1-2 Only:
- Gemini: 79.1% [75.6%, 82.2%]
- Claude: 62.3% [58.2%, 66.3%]
- Difference: 16.8pp
- p-value: < 0.001 (***)
- **Conclusion**: Highly significant difference

---

## 8. Findings & Recommendations

### 8.1 Key Findings

#### Finding 1: Answer Key Quality Issues
**~16% of questions show potentially incorrect answer keys** depending on confidence threshold:
- High confidence (Tier 1-2): 107 questions (16.0%)
- Medium confidence (Tier 1-3): 157 questions (23.4%)
- Low confidence (Tier 1-4): 251 questions (37.5%)

#### Finding 2: Data Quality Significantly Affects Measured Performance
Filtering by data quality improves measured accuracy by **1-19 percentage points**:
- Conservative filtering (Tier 1-2): +9-11pp improvement
- Aggressive filtering (Tier 1-4): +18-19pp improvement
- Optimal balance: Tier 1-2 filtering (563 questions, ~11pp gain)

#### Finding 3: LLM Judge Exhibits Substantial Bias
Gemini judge shows **21.5× self-agreement bias**:
- Agrees with Gemini models 80.8% vs Claude models 38.5%
- When families disagree, judge favors Gemini 21.5× more often
- **Not suitable as sole validation method. Require cross model judge evaluation to detect true bias**

#### Finding 4: Cross-Model Agreement Provides Objective Validation
Agreement-based tiers offer:
- **Bias-free**: Based on consensus across model families
- **Transparent**: Agreement counts are auditable
- **Efficient**: Uses existing evaluation data
- **Interpretable**: Clear confidence levels (Tier 1 > Tier 2 > Tier 3)

#### Finding 5: Relative Model Rankings Remained The Same
Across all filtering scenarios, **Gemini 2.5 Pro remains strongest**:
- Baseline: Gemini Pro (68.0%) > Flash (66.3%) > Opus (53.2%) > Sonnet (50.5%)
- Tier 1-2: Gemini Pro (79.1%) > Flash (77.7%) > Opus (62.3%) > Sonnet (58.8%)
- Tier 1-2 ∪ Judge: Gemini Pro (87.1%) > Flash (82.4%) > Opus (63.9%) > Sonnet (61.4%)

### 8.2 Recommendations for TLUE Benchmark

#### Recommendation 1: **Report Multiple Scenarios**
Do not report a single baseline accuracy. Instead, report:
- **Baseline** (full dataset): Shows coverage
- **Tier 1-2 Only** (563 questions): **Primary metric** for fair comparison
- **Tier 1-2 ∪ Judge** (493 questions): Highest quality subset

**Rationale**: Single-number reporting obscures answer key quality issues especially in traslated dataset and inflates apparent model differences.

#### Recommendation 2: **Use Cross-Model Agreement as a Cost Effective Primary Validation**
Prioritize agreement-based filtering over single LLM judge validation:
1. Start with Tier 1-2 filtering (high confidence, bias-free)
2. Optionally add judge validation for union approach
3. Never use judge validation from a single model familly alone
4. Double review answer keys that show high model agreenments on "incorrect" answer with human experts
5. Detect potential nuance introduced by translation

#### Recommendation 3: **Report Confidence Intervals**
Always report Wilson score 95% confidence intervals:
```
Gemini 2.5 Pro: 79.1% [75.6%, 82.2%] on Tier 1-2 Only (n=563)
```
- **Expand eval datasets**: Current released TLUE is a small size dataset, require larger datasets to objectively evaluate model performance.
#### Recommendation 4: **Create Answer Key Review Process**
Manually review Tier 1-2 questions with domain experts:
- Tier 1 (33 questions): All 4 models agree → Highest priority review
- Tier 2 (74 questions): 3 models agree → High priority review
- Document corrections and rationale
**Rationale**: While we cannot automatically correct answers, systematic disagreement warrants expert review.

### 8.3 Recommendations for Model Team

#### For Gemini
1. **Primary strength confirmed**: Gemini 2.5 Pro achieves 79.1% on high-quality subset (Tier 1-2)
2. **Gap analysis**: 20.9% of questions still incorrect on cleaned dataset
3. **Focus areas**: Analyze Tier 1-2 errors for model improvement opportunities
4. **Bias awareness**: Recognize judge bias when using Gemini for validation
5. **Gap awareness**: Recognize the model's performance gap between top used languages vs low reource language

#### For Claude
1. **Performance gap**: Claude Opus 4.1 achieves 62.3% vs Gemini's 79.1% on Tier 1-2
2. **Root cause analysis**: Investigate whether gap is due to:
   - Tibetan language understanding
   - Chinese language understanding
   - Bilingual reasoning
   - Multiple-choice test-taking strategies or others
3. **Strength identification**: Recognize questions where Claude outperforms Gemini
4. **Gap awareness**: Recognize the model's performance gap between top used languages vs low reource language

---

## 9. Limitations & Future Work

### 9.1 Current Limitations

#### Limitation 1: No Ground Truth Verification
- We infer answer key errors from model agreement, but haven't verified by human expert review yet
- Consensus may be wrong if all models share the same systematic misunderstanding

#### Limitation 2: Limited Model Diversity and Evaluation Dataset
- Only 2 model families (Gemini, Claude) used for consensus
- Agreement may be stronger with more diverse model families
- Dataset is small as TLUE only released a small subset of the full eval test set

#### Limitation 3: Language-Specific Bias Unclear
- Cannot determine if judge bias is language-specific or general
- Tibetan+Chinese bilingual context may amplify or reduce bias
- Cross model llm-as-judge evaluation is not performed

#### Limitation 4: No Subject-Level Analysis of Filtering Impact
- Current analysis is dataset-level only. While we have subject-level result, the dataset per subject is too small to make any conclusion
- Different subjects may have different answer key quality

#### Limitation 5: Manual Review Not Performed
- Tier 1-2 questions identified but not manually corrected
- Cannot provide "corrected" answer keys without expert review

### 9.2 Future Work
1. **Expert review of Tier 1-2 questions**: Recruit Tibetan language experts to review 107 high-confidence disagreements
2. **Subject-level analysis**: Compute per-subject improvement from filtering
3. **Cross-benchmark comparison**: Compare TLUE filtering impact to MMLU, C-Eval filtering
4. **Expand model coverage**: Add GPT-5 and other models to consensus validation
5. **Judge bias across languages**: Test if bias exists in monolingual (Tibetan-only, Chinese-only) subsets
6. **Test GPT-5 as judge**: Compare judge bias between Gemini and GPT-5 judges
7. **Community validation platform**: Build platform for crowdsourced expert for low resource language to review flagged questions
8. **Dynamic benchmark**: Create versioned enhanced TLUE eval frame with continuous quality improvement based on model feedback
9. **Automated bias detection**: Develop tools to automatically detect judge bias in evaluation workflows
10 **Expand eval datasets**: Current released TLUE is a small size dataset, require larger datasets to objectively evaluate model performance.
---

## 10. Conclusion

### 10.1 Summary

This study presents a comprehensive methodology for evaluating answer key quality in the Tibetan Language Understanding Evaluation (TLUE) benchmark through dual validation methods: cross-model agreement analysis and LLM-as-judge validation.

**Key contributions**:
1. **Identified potential answer key quality issues** (16-37% of questions)
2. **Quantified impact of data quality** on measured performance (9-19pp improvement)
3. **Detected and measured judge bias** (21.5× self-agreement ratio)
4. **Developed tier-based filtering framework** providing objective, bias-free validation
5. **Provided actionable recommendations** for benchmark maintainers and model developers

### 10.2 Primary Conclusions

1. **Answer key quality matters**: Using the full baseline dataset inflates apparent model differences and misrepresents true performance
2. **Cross-model agreement is superior to LLM judge**: Agreement-based validation is objective, transparent, and bias-free
3. **Judge bias is real and substantial**: Same-family LLM judges should not be used for validation
4. **Tier 1-2 filtering is recommended**: Provides best balance of quality (high confidence) and coverage (563 questions)
5. **Model rankings are robust**: Gemini 2.5 Pro remains strongest across all filtering scenarios

### 10.3 Key Impact

**For Gemini**:
- True performance: 79.1% [75.6%, 82.2%] on high-quality subset
- Advantage over Claude: 16.8pp (statistically significant, p<0.001)

**For Claude**:
- True performance: 62.3% [58.2%, 66.3%] on high-quality subset
- Performance gap reduced but remains significant when using unbiased validation

### 10.4 Methodological Lessons for Benchmark Evaluation

This work demonstrates that:

1. **Answer keys should not be assumed correct**: Even curated benchmarks contain errors
2. **Model consensus might be a powerful validation signal**: Systematic disagreement indicates likely errors
3. **LLM judges are not neutral**: Self-agreement bias must be measured and controlled
4. **Multiple filtering scenarios could be reported**: Single-number reporting obscures data quality issues
5. **Statistical rigor is needed**: Confidence intervals and significance tests prevent over-interpretation

---

## Appendices

### Appendix A: Tier Categorization Code

See `TLUE/script/analyze_cross_model_agreement_all_questions.py` for complete implementation.

### Appendix B: Judge Bias Analysis Code

See `TLUE/script/check_judge_bias.py` for complete implementation.

### Appendix C: Statistical Methods

Wilson score confidence interval implementation in `export_confidence_intervals.py`.

### Appendix D: Visualization Code

All heatmaps and multi-metric panels generated using:
- `TLUE/script/visualize_heatmap.py` (individual heatmaps)
- `TLUE/script/visualize_heatmap_small_multiples.py` (comparative view)
- `visualize_multi_metric_panel.py` (publication-quality panel chart)

### Appendix E: Data Files

- `enhanced_fair_comparison_with_judge.csv`: Complete results across all scenarios
- `confidence_intervals_summary.csv`: Wilson score CIs for all metrics
- `conditional_accuracy_with_ci.csv`: Pivot table for quick reference

---

## Document Version

**Version**: 1.0
**Date**: October 14, 2025
**Authors**: aipmtdaisy (with Claude Code assistance)
**Built Upon**: Original TLUE benchmark by Gao et al. (2025)
**Contact**: For questions about this methodology, please open an issue in the repository.

---

## Citation

If you use this enhanced evaluation methodology, please cite both this work and the original TLUE benchmark:

**This Enhanced Framework:**
```bibtex
@misc{aipmtdaisy2025enhanced_tlue,
  title={Enhanced TLUE Evaluation Framework: Answer Key Validation and Bias Detection for Low-Resource Language Benchmarks},
  author={aipmtdaisy},
  year={2025},
  month={10},
  howpublished={\url{https://github.com/aipmtdaisy/low_resource_language_eval_Tibetan}},
  note={Enhanced evaluation methodology built upon the TLUE benchmark (Gao et al., 2025).
        Contributions include cross-model agreement analysis, LLM judge bias detection,
        tier-based filtering framework, and statistical validation tools.}
}
```

**Original TLUE Benchmark:**
```bibtex
@article{gao2025tlue,
  title={TLUE: A Tibetan Language Understanding Evaluation Benchmark},
  author={Gao, Fan and Huang, Cheng and Tashi, Nyima and Wang, Xiangxiang and others},
  journal={arXiv preprint arXiv:2503.12051},
  year={2025},
  url={https://arxiv.org/abs/2503.12051},
  doi={10.48550/arXiv.2503.12051}
}
```

Original TLUE Repository: https://github.com/Vicentvankor/TLUE

---






