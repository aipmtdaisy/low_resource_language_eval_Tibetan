#!/usr/bin/env python3
"""
Calculate three levels of comparison metrics:
1. Full Dataset - All questions
2. Fair Comparison - Exclude questions where ANY model failed extraction
3. LLM-Validated Comparison - Exclude failed extractions AND questions with incorrect answer keys (validated by Gemini 2.5 Pro)

This ensures progressively stricter apples-to-apples comparison across different models.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List, Tuple

def load_truncated_questions(model_name: str) -> Set[str]:
    """Load the set of question IDs where answer extraction failed for a given model."""
    truncated_dir = Path(f"truncated_responses/{model_name}")
    truncated_locs = set()

    if not truncated_dir.exists():
        print(f"Warning: No truncated responses found for {model_name}")
        return truncated_locs

    # Read all truncated JSONL files for this model (with extracted_ prefix)
    for jsonl_file in truncated_dir.glob("extracted_*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    truncated_locs.add(data['loc'])

    return truncated_locs

def calculate_metrics_on_common_set(
    eval_file: Path,
    excluded_locs: Set[str]
) -> Tuple[int, int, int]:
    """
    Calculate metrics excluding questions in excluded_locs.
    Returns: (total_evaluated, valid_responses, correct_answers)
    """
    total_evaluated = 0
    valid_responses = 0
    correct_answers = 0

    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            loc = data.get('loc', '')

            # Skip questions that failed for any model
            if loc in excluded_locs:
                continue

            total_evaluated += 1
            answer = data.get('answer', '')
            model_answer = data.get('extracted_answer', [])

            # Check if model provided a valid answer
            if model_answer and len(model_answer) == 1:
                valid_responses += 1
                if model_answer[0] == answer:
                    correct_answers += 1

    return total_evaluated, valid_responses, correct_answers

def calculate_full_dataset_metrics(
    eval_file: Path
) -> Tuple[int, int, int]:
    """Calculate metrics on the full dataset without exclusions."""
    total_questions = 0
    valid_responses = 0
    correct_answers = 0

    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            total_questions += 1
            answer = data.get('answer', '')
            model_answer = data.get('extracted_answer', [])

            # Check if model provided a valid answer
            if model_answer and len(model_answer) == 1:
                valid_responses += 1
                if model_answer[0] == answer:
                    correct_answers += 1

    return total_questions, valid_responses, correct_answers

def load_answer_key_validation(validation_file: str) -> Dict[str, bool]:
    """
    Load LLM validation status for each question's answer key.

    Returns:
        Dict mapping loc → answer_key_correct (True/False/None)
    """
    validation_map = {}

    if not Path(validation_file).exists():
        print(f"Warning: LLM validation file not found: {validation_file}")
        return validation_map

    with open(validation_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                loc = entry.get('loc', '')
                val = entry.get('llm_validation', {})
                answer_key_correct = val.get('answer_key_correct')
                validation_map[loc] = answer_key_correct

    return validation_map

def calculate_llm_validated_metrics(
    combined_file: Path,
    validation_map: Dict[str, bool]
) -> Tuple[int, int, int, int, int, int, int]:
    """
    Calculate 4-tier accuracy metrics using LLM validation.

    Returns: (total, extracted, correct, extracted_valid, correct_valid, invalid_count, unvalidated_count)
    """
    total = 0
    extracted = 0
    correct = 0
    extracted_valid = 0  # Extracted on valid answer keys (true or null)
    correct_valid = 0    # Correct on valid answer keys
    invalid_count = 0    # Questions with answer_key_correct = false
    unvalidated_count = 0  # Questions with answer_key_correct = null

    with open(combined_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            entry = json.loads(line)
            loc = entry.get('loc', '')
            total += 1

            answer = entry.get('answer', '')
            model_answer = entry.get('extracted_answer', [])
            has_extracted = bool(model_answer and len(model_answer) == 1)

            if has_extracted:
                extracted += 1
                if model_answer[0] == answer:
                    correct += 1

            # Check validation status
            answer_key_status = validation_map.get(loc, None)

            if answer_key_status == False:
                invalid_count += 1
            elif answer_key_status is None:
                unvalidated_count += 1

            # Count on valid answer keys (true or null, excluding false)
            if answer_key_status != False:
                if has_extracted:
                    extracted_valid += 1
                    if model_answer[0] == answer:
                        correct_valid += 1

    return total, extracted, correct, extracted_valid, correct_valid, invalid_count, unvalidated_count

def main():
    # Models to compare (new round + previous round; gemini-3-pro & gemini-3-1-pro excluded pending re-run)
    models = [
        'claude-opus-4-6', 'claude-opus-4-5', 'claude-sonnet-4-6', 'claude-sonnet-4-5',
        'claude-opus-4-1',
        'gemini-3-flash', 'gemini-2-5-flash', 'gemini-2-5-pro',
    ]

    print("="*120)
    print("Model Evaluation Metrics - Full Dataset, Fair Comparison, and LLM-Validated Comparison")
    print("="*120)
    print()

    # PART 1: Full Dataset Metrics
    print("PART 1: Full Dataset Metrics (All 1340 questions)")
    print("-" * 120)

    full_results = []
    for model in models:
        eval_dir = Path(f"TLUE/model_answer/{model}_eval_res")
        if not eval_dir.exists():
            continue

        total_questions = 0
        total_valid = 0
        total_correct = 0

        for jsonl_file in eval_dir.glob("extracted_*.jsonl"):
            questions, valid, correct = calculate_full_dataset_metrics(jsonl_file)
            total_questions += questions
            total_valid += valid
            total_correct += correct

        response_rate = (total_valid / total_questions * 100) if total_questions > 0 else 0
        accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        conditional_accuracy = (total_correct / total_valid * 100) if total_valid > 0 else 0

        full_results.append({
            'model': model,
            'total_questions': total_questions,
            'valid_responses': total_valid,
            'correct_answers': total_correct,
            'response_rate': response_rate,
            'accuracy': accuracy,
            'conditional_accuracy': conditional_accuracy
        })

    # Print full dataset results
    print(f"{'Model':<25} {'Total':<10} {'Valid Resp':<12} {'Correct':<10} {'Resp Rate':<12} {'Accuracy':<12} {'Cond. Acc':<12}")
    print("-" * 120)
    for r in full_results:
        print(f"{r['model']:<25} {r['total_questions']:<10} {r['valid_responses']:<12} "
              f"{r['correct_answers']:<10} {r['response_rate']:>10.2f}% {r['accuracy']:>10.2f}% "
              f"{r['conditional_accuracy']:>10.2f}%")
    print("-" * 120)
    print()

    # PART 2: Fair Comparison (Common Question Set)
    print("PART 2: Fair Comparison - Common Question Set (Excluding failed extractions)")
    print("-" * 120)

    # Step 1: Collect all truncated questions across all models
    all_excluded_locs = set()
    model_truncated_counts = {}

    for model in models:
        truncated_locs = load_truncated_questions(model)
        model_truncated_counts[model] = len(truncated_locs)
        all_excluded_locs.update(truncated_locs)

    print(f"Questions excluded per model:")
    for model, count in model_truncated_counts.items():
        print(f"  {model}: {count} failed extractions")
    print(f"\nTotal unique questions excluded: {len(all_excluded_locs)}")
    print()

    # Step 2: Calculate metrics for each model on the common question set
    results = []

    for model in models:
        # Find the evaluation result file
        eval_dir = Path(f"TLUE/model_answer/{model}_eval_res")

        if not eval_dir.exists():
            print(f"Warning: Evaluation results not found for {model}")
            continue

        # Aggregate across all subject files
        total_evaluated = 0
        total_valid = 0
        total_correct = 0

        for jsonl_file in eval_dir.glob("extracted_*.jsonl"):
            evaluated, valid, correct = calculate_metrics_on_common_set(
                jsonl_file, all_excluded_locs
            )
            total_evaluated += evaluated
            total_valid += valid
            total_correct += correct

        # Calculate metrics
        response_rate = (total_valid / total_evaluated * 100) if total_evaluated > 0 else 0
        accuracy = (total_correct / total_evaluated * 100) if total_evaluated > 0 else 0
        conditional_accuracy = (total_correct / total_valid * 100) if total_valid > 0 else 0

        results.append({
            'model': model,
            'total_evaluated': total_evaluated,
            'valid_responses': total_valid,
            'correct_answers': total_correct,
            'response_rate': response_rate,
            'accuracy': accuracy,
            'conditional_accuracy': conditional_accuracy
        })

    # Print fair comparison results
    print(f"{'Model':<25} {'Evaluated':<12} {'Valid Resp':<12} {'Correct':<10} {'Resp Rate':<12} {'Accuracy':<12} {'Cond. Acc':<12}")
    print("-" * 120)

    for r in results:
        print(f"{r['model']:<25} {r['total_evaluated']:<12} {r['valid_responses']:<12} "
              f"{r['correct_answers']:<10} {r['response_rate']:>10.2f}% {r['accuracy']:>10.2f}% "
              f"{r['conditional_accuracy']:>10.2f}%")

    print("-" * 120)
    print()

    # PART 3: LLM-Validated 4-Tier Accuracy Metrics
    print("PART 3: LLM-Validated 4-Tier Accuracy Metrics (Full Dataset with Quality Indicators)")
    print("-" * 120)

    # Load LLM validation results
    validation_file = "TLUE/model_answer/gemini-3-pro_eval_res/gemini-3-pro_llm_validated_v3_retry.jsonl"
    validation_map = load_answer_key_validation(validation_file)

    if len(validation_map) > 0:
        # Count validation status
        valid_count = sum(1 for v in validation_map.values() if v == True)
        invalid_count = sum(1 for v in validation_map.values() if v == False)
        unvalidated_count = sum(1 for v in validation_map.values() if v is None)

        print(f"Answer Key Validation Status:")
        print(f"  ✓ Correct: {valid_count} questions")
        print(f"  ✗ Incorrect: {invalid_count} questions")
        print(f"  ? Unvalidated: {unvalidated_count} questions")
        print()

        # Calculate 4-tier metrics for each model
        llm_results = []

        for model in models:
            combined_file = Path(f"TLUE/model_answer/{model}_eval_res/{model}_combined_results.jsonl")

            if not combined_file.exists():
                print(f"Warning: Combined results not found for {model}")
                continue

            total, extracted, correct, extracted_valid, correct_valid, inv_cnt, unval_cnt = \
                calculate_llm_validated_metrics(combined_file, validation_map)

            # Calculate 4 metrics
            response_rate = (extracted / total * 100) if total > 0 else 0
            full_accuracy = (correct / total * 100) if total > 0 else 0
            extracted_accuracy = (correct / extracted * 100) if extracted > 0 else 0
            actual_accuracy = (correct_valid / extracted_valid * 100) if extracted_valid > 0 else 0

            llm_results.append({
                'model': model,
                'response_rate': response_rate,
                'full_accuracy': full_accuracy,
                'extracted_accuracy': extracted_accuracy,
                'actual_accuracy': actual_accuracy
            })

        # Print 4-tier metrics table
        print(f"{'Model':<25} {'Response Rate':<15} {'Full Accuracy':<15} {'Extracted Acc':<15} {'Actual Acc':<15}")
        print("-" * 120)

        for r in llm_results:
            print(f"{r['model']:<25} {r['response_rate']:>13.2f}% {r['full_accuracy']:>13.2f}% "
                  f"{r['extracted_accuracy']:>13.2f}% {r['actual_accuracy']:>13.2f}%")

        print("-" * 120)
        print()
        print("Metrics explained:")
        print("  - Response Rate: Questions with extracted answer / Total questions")
        print("  - Full Accuracy: Correct answers / Total questions (penalizes extraction failures)")
        print("  - Extracted Accuracy: Correct / Extracted (accuracy when model answers)")
        print("  - Actual Accuracy: Correct / Extracted on valid answer keys (excludes bad answer keys)")
        print()
    else:
        print("LLM validation data not available. Skipping Part 3.")
        print()
        llm_results = []

    # Save both full and fair comparison to CSV
    full_output_file = "full_dataset_metrics.csv"
    with open(full_output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model', 'total_questions', 'valid_responses', 'correct_answers',
            'response_rate', 'accuracy', 'conditional_accuracy'
        ])
        writer.writeheader()
        writer.writerows(full_results)

    fair_output_file = "fair_comparison_metrics.csv"
    with open(fair_output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model', 'total_evaluated', 'valid_responses', 'correct_answers',
            'response_rate', 'accuracy', 'conditional_accuracy'
        ])
        writer.writeheader()
        writer.writerows(results)

    # Save LLM-validated 4-tier metrics if available
    if llm_results:
        llm_output_file = "llm_validated_4tier_metrics.csv"
        with open(llm_output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'model', 'response_rate', 'full_accuracy', 'extracted_accuracy', 'actual_accuracy'
            ])
            writer.writeheader()
            writer.writerows(llm_results)

        print(f"Full dataset results saved to: {full_output_file}")
        print(f"Fair comparison results saved to: {fair_output_file}")
        print(f"LLM-validated 4-tier metrics saved to: {llm_output_file}\n")
    else:
        print(f"Full dataset results saved to: {full_output_file}")
        print(f"Fair comparison results saved to: {fair_output_file}\n")

if __name__ == "__main__":
    main()
