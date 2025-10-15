# Enhance eval tool built on top of TLUE: Tibetan Language Understanding Evaluation

A comprehensive benchmark frame for evaluating large language models on Tibetan multiple-choice questions with rigorous answer key validation methodology and model bias detection tool.

## Acknowledgments

This project builds upon the TLUE (Tibetan Language Understanding Evaluation) benchmark. If you use this repository, please cite the original TLUE paper:
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

## Overview

This eval frame is built on top of the original TLUE (Tibetan Language Understanding Evaluation) is a research project that evaluates state-of-the-art large language models on their understanding of Tibetan-Chinese bilingual content. The partially realeased Ti-MMLU subset contains **670 questions** across **67 academic subjects**, ranging from agronomy to world history.

### What's New in This Enhanced Version

This repository extends the [original TLUE benchmark](https://github.com/Vicentvankor/TLUE) with comprehensive answer key validation and statistical rigor:

**Original TLUE provides:**
- Ti-MMLU dataset (670 questions, 67 subjects)
- Basic model evaluation framework
- Direct and comprehensive answer extraction methods
- Support for Tibetan and English option recognition etc.

**The enhancements added:**
- **Answer Key Validation** (16-37% of questions flagged as potentially incorrect)
- **Cross-Model Agreement Analysis** (consensus validation across 4 models)
- **LLM-as-Judge Validation** with systematic bias detection (21.5× self-agreement ratio quantified)
- **Statistical Rigor** (Wilson score confidence intervals, two-proportion z-tests)
- **Tier-Based Filtering Framework** (5 confidence levels for data quality assessment)
- **Enhanced error handling Code** (error handling, input validation, unit tests)
- **14 Visualization Scripts** (heatmaps, radar charts, multi-metric panels, judge bias analysis)
- **5 Utility Modules** (error_handler, input_validator, llm_judge_utils, config_loader, visualization_utils)
- **5 Unit Test Suites** (unit test coverage for all utilities)

This repository includes:
- Complete evaluation framework for LLM testing
- Cross-model agreement analysis for answer key validation
- LLM-as-judge validation with bias detection
- Statistical analysis tools with confidence intervals
- Visualization utilities for results analysis

## Key Features

- **Rigorous Validation**: Dual-method answer key validation (cross-model agreement + LLM-as-judge)
- **Bias Detection**: Quantified judge bias (21.5× self-agreement ratio) with mitigation strategies (human in the loop recommended to further investigate potential bias)
- **Statistical Rigor**: Wilson score confidence intervals and significance testing
- **Multi-Model Support**: Tested with Gemini 2.5 Pro/Flash and Claude Opus 4.1/Sonnet 4.5
- **Tier-Based Filtering**: Confidence-based question filtering (16-37% of questions flagged) to reduce potential answer key quality issue 

## Quick Start

### Prerequisites

```bash
pip install -r TLUE/requirements.txt
```

For development dependencies:
```bash
pip install -r TLUE/requirements-dev.txt
```

### Configuration

1. Copy the example environment file:
```bash
cp TLUE/.env.example TLUE/.env
```

2. Add your API keys to `TLUE/.env`:
```
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
```

3. Verify configuration:
```bash
cd TLUE/script
python test_config.py
```

### Running Evaluations

1. **Run models on the dataset**:
```bash
cd TLUE/script
python run_models.py
```

2. **Evaluate results**:
```bash
python auto_evaluate.py <model_id>
```

3. **Analyze cross-model agreement**:
```bash
python analyze_cross_model_agreement_all_questions.py
```

## Evaluation Results

### Baseline Performance (670 questions)

| Model | Conditional Accuracy |
|-------|---------------------|
| Gemini 2.5 Pro | 68.0% [64.4%, 71.4%] |
| Gemini 2.5 Flash | 66.3% |
| Claude Opus 4.1 | 53.2% [49.4%, 57.0%] |
| Claude Sonnet 4.5 | 50.5% |

### Tier 1-2 Filtered (563 questions, recommended)

| Model | Conditional Accuracy |
|-------|---------------------|
| Gemini 2.5 Pro | **79.1% [75.6%, 82.2%]** |
| Gemini 2.5 Flash | 77.7% |
| Claude Opus 4.1 | 62.3% [58.2%, 66.3%] |
| Claude Sonnet 4.5 | 58.8% |

**Filtering improves measured accuracy by 9-19 percentage points** by excluding questions with likely answer key errors (more than 3 models agreed on the same answer which is different from the provided answer key).

## Methodology

Our validation methodology includes:

1. **Cross-Model Agreement Analysis**: Identifies questions where multiple independent models agree on an answer different from the provided key
2. **LLM-as-Judge Validation**: Uses Gemini 2.5 Pro to verify answer keys with bias detection
3. **Tier-Based Filtering**: Confidence levels from Tier 1 (all 4 models agree) to Tier 5 (no consensus)
4. **Statistical Validation**: Wilson score confidence intervals and two-proportion z-tests

For complete methodology details, see [METHODOLOGY.md](METHODOLOGY.md).

## Repository Structure

```
TLUE/
├── data/                           # Dataset files (Ti-MMLU subset)
├── script/                         # Main evaluation scripts
│   ├── run_models.py              # Run models on dataset
│   ├── auto_evaluate.py           # Evaluate model outputs
│   ├── analyze_cross_model_agreement_all_questions.py
│   ├── check_judge_bias.py        # Detect judge bias
│   ├── visualize_*.py             # Visualization tools
│   └── test_*.py                  # Unit tests (5 test files)
├── model_answer/                   # Model evaluation results
├── logs/                          # Execution logs
├── requirements.txt               # Python dependencies
├── requirements-dev.txt           # Development dependencies
├── .env.example                   # Environment variable template
└── config.yaml                    # Main configuration file

Root/
├── METHODOLOGY.md                 # Detailed methodology documentation
└── README.md                      # This file
```

## Testing

The repository includes comprehensive unit tests:

```bash
cd TLUE/script

# Test configuration
python test_config.py

# Test error handling
python test_error_handling.py

# Test input validation
python test_input_validation.py

# Test LLM judge utilities
python test_llm_judge_utils.py

# Test visualization utilities
python test_visualization_utils.py
```

## Key Findings

1. **16-37% of questions have potentially incorrect answer keys** (depending on confidence threshold)
2. **Data quality significantly affects measured performance** (+9 to +19 percentage points improvement)
3. **LLM judges exhibit substantial bias** (21.5× self-agreement ratio for Gemini judge) or Gemini model as a judge indeed show much higher accuracy (require human validation)
4. **Cross-model agreement provides objective validation** (bias-free, transparent, efficient)
5. **Model rankings are preserved across filtering scenarios**

## Citation

### Citing This Enhanced Framework

If you use our enhanced evaluation methodology (answer key validation, bias detection, tier-based filtering, or statistical tools), please cite:

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

### Citing the Original TLUE Benchmark

This project builds upon the TLUE (Tibetan Language Understanding Evaluation) benchmark. If you use the Ti-MMLU dataset or the original TLUE framework, please also cite:

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

## License

This original TLUE project is released under the MIT License. The methodology document is released under CC-BY 4.0.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Security

This repository includes:
- Input validation for file paths and model IDs
- Environment variable-based API key management
- Comprehensive error handling and retry logic

See [TLUE/SECURITY_IMPROVEMENTS.md](TLUE/SECURITY_IMPROVEMENTS.md) for details.

## Contact

For questions about this evaluation framework or methodology, please open an issue in this repository.

## Acknowledgments

### Original TLUE Benchmark

This project builds upon the **TLUE (Tibetan Language Understanding Evaluation)** benchmark created by Gao et al. We are grateful for their foundational work in creating a comprehensive evaluation framework for Tibetan language understanding.

**Original TLUE contributions:**
- Ti-MMLU dataset (670 questions across 67 subjects)
- Bilingual Tibetan-Chinese question format
- Basic model evaluation framework
- Answer extraction methods

Original repository: [https://github.com/Vicentvankor/TLUE](https://github.com/Vicentvankor/TLUE)

### How to Cite

When using this repository:
- **If using the enhanced methodology** (answer key validation, bias detection, filtering): Cite both the enhanced framework and original TLUE (see [Citation](#citation) section)
- **If using only the Ti-MMLU dataset and/or orginal TLUE repo**: Cite the original TLUE paper



