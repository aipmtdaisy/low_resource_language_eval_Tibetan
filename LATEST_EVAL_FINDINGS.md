# TLUE Evaluation Findings — 8-Model Comparison

**Date**: March 2, 2026
**Dataset**: 670 questions across 67 subjects (Tibetan Language Understanding Evaluation)
**Models**: 8 LLMs (3 Gemini, 5 Claude)

---

## Progress from Original TLUE Paper

The original TLUE paper (Gao et al., 2025) evaluated 13 LLMs on Ti-MMLU, with most models scoring near or below the **25% random baseline** (4-choice MC). Our evaluation on the same benchmark type but with a small released datasets. Overall, it shows dramatic improvement across model families.

| Model (Original Paper) | Ti-MMLU CAA (full dataset)| Model (This Eval-670) | Baseline CAA (670)| Tier 1-2 CAA (670) |
|---|---|---|---|---|
| Claude-3.5-Sonnet | 35.6% | Claude Opus 4.6 | 59.5% | 64.8% |
| Gemini 1.5-Flash | 31.0% | Gemini 3 Flash | 73.9% | 79.0% |
| DeepSeek-V3 | 32.2% | — | — | — |
| GPT-4o | 17.5% | — | — | — |

*(CAA — same as our conditional accuracy metric)*

Both model families have improved significantly. Gemini has made the largest leap: **+42.9pp** (31.0% → 73.9%), overtaking Claude which was previously the top performer at 35.6%. Claude Opus 4.6 improved **+23.9pp** over Claude-3.5-Sonnet. Gemini went from trailing Claude to leading by ~14pp.

---

## Models Evaluated

| Model | Provider | Family |
|-------|----------|--------|
| Gemini 3 Flash | Google | Gemini |
| Gemini 2.5 Pro | Google | Gemini |
| Gemini 2.5 Flash | Google | Gemini |
| Claude Opus 4.6 | Anthropic | Claude |
| Claude Opus 4.5 | Anthropic | Claude |
| Claude Opus 4.1 | Anthropic | Claude |
| Claude Sonnet 4.6 | Anthropic | Claude |
| Claude Sonnet 4.5 | Anthropic | Claude |

---

## Baseline Performance (670 Questions)

Ranked by conditional accuracy (accuracy among valid responses):

| Rank | Model | Valid Responses | Response Rate | Conditional Accuracy |
|------|-------|----------------|---------------|---------------------|
| 1 | Gemini 3 Flash | 662/670 | 98.8% | **73.9%** |
| 2 | Gemini 2.5 Pro | 657/670 | 98.1% | **68.0%** |
| 3 | Gemini 2.5 Flash | 649/670 | 96.9% | **66.3%** |
| 4 | Claude Opus 4.6 | 619/670 | 92.4% | **59.5%** |
| 5 | Claude Opus 4.5 | 633/670 | 94.5% | **58.5%** |
| 6 | Claude Opus 4.1 | 643/670 | 96.0% | **53.2%** |
| 7 | Claude Sonnet 4.5 | 567/670 | 84.6% | **49.0%** |
| 8 | Claude Sonnet 4.6 | 658/670 | 98.2% | **45.6%** |

---

## Tier-Based Filtered Performance

Filtering removes questions with likely answer key errors (identified by cross-model agreement). See [Answer Key Quality](#answer-key-quality-summary) below for tier definitions.

### Tier 1-2 Filtered — 617 Questions (Recommended)

| Rank | Model | Conditional Accuracy | Change from Baseline |
|------|-------|---------------------|---------------------|
| 1 | Gemini 3 Flash | **79.0%** | +5.1pp |
| 2 | Gemini 2.5 Pro | **73.2%** | +5.1pp |
| 3 | Gemini 2.5 Flash | **71.8%** | +5.6pp |
| 4 | Claude Opus 4.6 | **64.8%** | +5.4pp |
| 5 | Claude Opus 4.5 | **63.4%** | +5.0pp |
| 6 | Claude Opus 4.1 | **57.5%** | +4.3pp |
| 7 | Claude Sonnet 4.5 | **53.5%** | +4.5pp |
| 8 | Claude Sonnet 4.6 | **49.6%** | +4.0pp |

### Tier 1-3 Filtered — 500 Questions

| Rank | Model | Conditional Accuracy | Change from Baseline |
|------|-------|---------------------|---------------------|
| 1 | Gemini 3 Flash | **86.4%** | +12.5pp |
| 2 | Gemini 2.5 Pro | **83.6%** | +15.5pp |
| 3 | Gemini 2.5 Flash | **81.4%** | +15.2pp |
| 4 | Claude Opus 4.6 | **71.5%** | +12.0pp |
| 5 | Claude Opus 4.5 | **70.8%** | +12.4pp |
| 6 | Claude Opus 4.1 | **61.3%** | +8.1pp |
| 7 | Claude Sonnet 4.5 | **59.4%** | +10.4pp |
| 8 | Claude Sonnet 4.6 | **55.7%** | +10.1pp |

### Tier 1-4 Filtered — 378 Questions

| Rank | Model | Conditional Accuracy | Change from Baseline |
|------|-------|---------------------|---------------------|
| 1 | Gemini 3 Flash | **85.4%** | +11.5pp |
| 2 | Gemini 2.5 Pro | **82.5%** | +14.4pp |
| 3 | Gemini 2.5 Flash | **80.7%** | +14.4pp |
| 4 | Claude Opus 4.6 | **79.8%** | +20.4pp |
| 5 | Claude Opus 4.5 | **79.5%** | +21.0pp |
| 6 | Claude Opus 4.1 | **71.5%** | +18.3pp |
| 7 | Claude Sonnet 4.5 | **71.3%** | +22.2pp |
| 8 | Claude Sonnet 4.6 | **65.0%** | +19.4pp |

---

## Answer Key Quality Summary

Cross-model agreement analysis flags questions where multiple models agree on an answer that differs from the provided key:

| Tier | Criteria | Questions | % of Dataset |
|------|----------|-----------|-------------|
| Tier 1 (Highest confidence) | All 8 models agree, differs from key | 36 | 5.4% |
| Tier 2 (High confidence) | 7+ models agree, differs from key | 17 | 2.5% |
| Tier 3 (Medium confidence) | Cross-brand majority agrees, differs from key | 117 | 17.5% |
| Tier 4 (Low confidence) | Same-brand majority only, differs from key | 122 | 18.2% |

**Cumulative**: Tier 1-2 = 53 questions (7.9%), Tier 1-3 = 170 (25.4%), Tier 1-4 = 292 (43.6%)

---

## Key Takeaways

1. **Massive progress since the original TLUE paper** — Gemini 3 Flash (73.9%) more than doubled Gemini 1.5-Flash's original 31.0%, and Claude Opus 4.6 (59.5%) improved +23.9pp over Claude-3.5-Sonnet's 35.6%. **Gemini went from trailing Claude to leading by ~14pp.**

2. **Gemini 3 Flash leads across all scenarios** — 73.9% baseline, 79.0% on the recommended Tier 1-2 subset (617 questions)

3. **Answer key quality significantly impacts results** — filtering likely errors improves all models by 4-22pp, with Claude models benefiting most at aggressive thresholds

4. **Relative model rankings are stable** — the ordering Gemini 3 Flash > 2.5 Pro > 2.5 Flash > Claude Opus 4.6 > 4.5 > 4.1 > Sonnet 4.5 > Sonnet 4.6 holds across all filtering scenarios

5. **~8% of questions have high-confidence answer key issues** — 53 questions (Tier 1-2) where 7-8 models unanimously disagree with the provided key

6. **Newer Claude Opus models improve over 4.1** — Opus 4.6 (59.5%) and 4.5 (58.5%) both surpass Opus 4.1 (53.2%) at baseline, though a ~14pp gap with Gemini 3 Flash persists on the Tier 1-2 subset

7. **Response rates are generally high** (84.6-98.8%), with Claude Sonnet 4.5 as the outlier at 84.6%

---

## Recommendations

- **Report Tier 1-2 filtered results (617 Q) as the primary metric** — this excludes only the highest-confidence answer key errors while retaining 92% of the dataset
- **Always include baseline (670 Q) for transparency** — shows the unfiltered picture alongside the filtered one
- **Use cross-model agreement over single-judge validation** — agreement-based tiers are bias-free and auditable
- **Report Wilson score 95% confidence intervals** — e.g., Gemini 3 Flash Tier 1-2: 79.0% [75.6%, 82.0%]
- **Prioritize expert review of Tier 1 (36 Q) and Tier 2 (17 Q)** — these 53 questions have the strongest evidence of answer key error
- **Expand the evaluation dataset** — the current 670-question TLUE release is small; larger datasets would strengthen conclusions

---

## References

- **Full methodology**: See [METHODOLOGY.md](METHODOLOGY.md) for tier definitions, judge bias analysis, statistical methods, and code pointers
- **Original TLUE benchmark**: Gao et al. (2025), [arXiv:2503.12051](https://arxiv.org/abs/2503.12051)
- **Original TLUE repository**: https://github.com/Vicentvankor/TLUE
