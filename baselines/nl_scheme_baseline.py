#!/usr/bin/env python3
"""
NL-Only Baseline: Scheme-Guided Premise Reconstruction.

Given a premise, hypothesis, and argumentation scheme (in natural language),
prompts the LLM to predict the missing premise. No formalisation or Isabelle.

Usage:
    python baselines/nl_scheme_baseline.py --llm Qwen/Qwen2.5-3B-Instruct \
        --data_name nlas-multi-preprocessed --run_id nl_baseline_test
"""

import json
import os
import sys
import argparse
from datetime import datetime

import yaml
import torch.cuda as cuda

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from generation.local_llm import LocalLLM


def load_scheme_descriptions(config_path='config.yaml'):
    """Load scheme name -> description mapping from config.yaml."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    schemes = config.get('walton_argumentation_schemes', {}).get('schemes', [])
    return {s['name']: s['description'] for s in schemes}


def main(args):
    # Load data
    data_path = os.path.join('data', f'{args.data_name}.json')
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Resolve output directory
    if args.run_id is None:
        run_id = f'nl_baseline_{args.data_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    else:
        run_id = args.run_id
    output_dir = os.path.join('results', run_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results directory: {output_dir}")

    # Load scheme descriptions
    scheme_descriptions = load_scheme_descriptions()
    print(f"Loaded {len(scheme_descriptions)} scheme descriptions")

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
        scheme_name = item.get('argumentation_scheme', '')
        scheme_description = scheme_descriptions.get(scheme_name, '')

        if not scheme_description:
            print(f"Warning: no description for scheme '{scheme_name}', using name only")
            scheme_description = scheme_name

        print(f"Processing {data_name} (scheme: {scheme_name})")

        # Generate missing premise prediction
        predicted_premise = llm.generate(
            model_prompt_dir='nl_baseline',
            prompt_name='scheme_premise_prompt.txt',
            scheme_name=scheme_name,
            scheme_description=scheme_description,
            premise=premise,
            hypothesis=hypothesis,
        )

        # Build result (compatible with analyze_premise_results.py)
        result = dict(item)
        result['results'] = {
            'semantic validity': None,
            'predicted_premise': predicted_premise.strip(),
            'method': 'nl_scheme_baseline',
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
        description='NL-only baseline: scheme-guided premise reconstruction'
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
