#!/usr/bin/env python3
"""
Extract questions with truncated or empty responses from Claude model evaluation results.
A response is considered truncated if filtered_model_result is empty.
"""

import json
import os
from pathlib import Path

def extract_truncated_responses(input_dir, output_dir, model_name):
    """Extract questions with empty filtered_model_result (truncated responses)."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Track statistics
    total_questions = 0
    truncated_questions = 0
    truncated_by_subject = {}

    # Process each JSONL file
    for jsonl_file in input_path.glob("*.jsonl"):
        # Extract subject name from filename
        filename = jsonl_file.stem  # removes .jsonl extension

        truncated_entries = []
        file_total = 0
        file_truncated = 0

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    file_total += 1
                    total_questions += 1

                    try:
                        entry = json.loads(line)

                        # Check if filtered_model_result OR direct_answer is empty
                        # Different file types use different field names
                        filtered_result = entry.get('filtered_model_result')
                        direct_answer = entry.get('direct_answer')

                        # Consider truncated if the answer field exists but is empty
                        is_truncated = False
                        if filtered_result is not None and not filtered_result:
                            is_truncated = True
                        elif direct_answer is not None and not direct_answer:
                            is_truncated = True

                        if is_truncated:
                            truncated_entries.append(entry)
                            file_truncated += 1
                            truncated_questions += 1
                    except json.JSONDecodeError:
                        print(f"Error parsing line in {jsonl_file}: {line[:100]}")

        # Save truncated entries for this file
        if truncated_entries:
            output_file = output_path / f"{filename}_truncated.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for entry in truncated_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            truncated_by_subject[filename] = file_truncated
            print(f"✓ {filename}: {file_truncated}/{file_total} truncated")

    # Save summary
    summary = {
        "model": model_name,
        "total_questions": total_questions,
        "truncated_questions": truncated_questions,
        "truncation_rate": f"{truncated_questions/total_questions*100:.2f}%" if total_questions > 0 else "0%",
        "by_subject": truncated_by_subject
    }

    summary_file = output_path / f"{model_name}_truncated_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Total questions: {total_questions}")
    print(f"Truncated questions: {truncated_questions}")
    print(f"Truncation rate: {summary['truncation_rate']}")
    print(f"{'='*60}\n")

    return summary

def main():
    # Use relative paths from script location
    script_dir = Path(__file__).parent
    base_dir = script_dir / "TLUE" / "model_answer"
    output_base = script_dir / "truncated_responses"

    # Process all models
    models = [
        {
            "name": "claude-sonnet-4-5",
            "dir": base_dir / "claude-sonnet-4-5_eval_res"
        },
        {
            "name": "claude-opus-4-1",
            "dir": base_dir / "claude-opus-4-1_eval_res"
        },
        {
            "name": "gemini-2-5-pro",
            "dir": base_dir / "gemini-2-5-pro_eval_res"
        },
        {
            "name": "gemini-2-5-flash",
            "dir": base_dir / "gemini-2-5-flash_eval_res"
        }
    ]

    all_summaries = []

    for model in models:
        print(f"\nProcessing {model['name']}...")
        print("-" * 60)

        output_dir = output_base / model['name']
        summary = extract_truncated_responses(
            model['dir'],
            output_dir,
            model['name']
        )
        all_summaries.append(summary)

    # Save combined summary
    combined_summary_file = output_base / "combined_truncation_summary.json"
    with open(combined_summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print(f"\n✅ All truncated responses extracted to: {output_base}")
    print(f"✅ Combined summary saved to: {combined_summary_file}")

if __name__ == "__main__":
    main()
