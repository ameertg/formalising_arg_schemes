#!/usr/bin/env python3
"""
NL No-Scheme Baseline: Premise Reconstruction Without Scheme Knowledge.

Given a premise and hypothesis (no argumentation scheme info),
prompts the LLM to predict the missing premise. Ablation of nl_scheme_baseline.

Usage:
    python baselines/nl_no_scheme_baseline.py --llm Qwen/Qwen2.5-3B-Instruct \
        --data_name nlas-multi-preprocessed --run_id nl_no_scheme_test
"""

import json
import os
import sys
import argparse
from datetime import datetime

import torch.cuda as cuda

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from generation.local_llm import LocalLLM


def main(args):
    # Load data
    data_path = os.path.join('data', f'{args.data_name}.json')
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Resolve output directory
    if args.run_id is None:
        run_id = f'nl_no_scheme_{args.data_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    else:
        run_id = args.run_id
    output_dir = os.path.join('results', run_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results directory: {output_dir}")

    # Initialise LLM
    model_id = args.llm
    device = 'cuda' if cuda.is_available() else 'mps'
    print(f"Using local model: {model_id} on device: {device}")
    llm = LocalLLM(model_id, device=device, load_in_8bit=args.load_in_8bit)

    for item in data:
        q_id = item['id']
        data_name = f'{args.data_name}_{q_id}'

        # Checkpoint: skip if result already exists
        result_path = os.path.join(output_dir, f'{data_name}_results.json')
        if os.path.exists(result_path):
            print(f"Skipping {data_name}: checkpoint exists")
            continue

        premise = item['premise']
        hypothesis = item['hypothesis']

        print(f"Processing {data_name}")

        # Generate missing premise prediction (no scheme info)
        predicted_premise = llm.generate(
            model_prompt_dir='nl_baseline',
            prompt_name='no_scheme_premise_prompt.txt',
            premise=premise,
            hypothesis=hypothesis,
        )

        # Build result (compatible with analyze_premise_results.py)
        result = dict(item)
        result['results'] = {
            'semantic validity': None,
            'predicted_premise': predicted_premise.strip(),
            'method': 'nl_no_scheme_baseline',
        }
        result['ground_truth_premise'] = item.get('removed_premise', '')
        result['ground_truth_premise_type'] = item.get('removed_premise_type', '')
        result['predicted_premises'] = predicted_premise.strip()

        with open(result_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"  Predicted: {predicted_premise.strip()[:120]}...")

    print(f"\nDone. Results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NL no-scheme baseline: premise reconstruction without scheme knowledge'
    )
    parser.add_argument('--llm', '-l', type=str, required=True,
                        help='HuggingFace model ID (e.g. Qwen/Qwen2.5-3B-Instruct)')
    parser.add_argument('--data_name', '-d', type=str,
                        default='nlas-multi-preprocessed',
                        help='Dataset name (default: nlas-multi-preprocessed)')
    parser.add_argument('--load_in_8bit', action='store_true',
                        help='Load model in 8-bit quantization (CUDA only)')
    parser.add_argument('--run_id', '-r', type=str, default=None,
                        help='Run identifier for checkpointing')
    args = parser.parse_args()
    main(args)
