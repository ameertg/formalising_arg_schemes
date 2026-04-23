from refinement.refinement_model import RefinementModel
from generation.gpt import GPT
from generation.local_llm import LocalLLM
from critique.isabelle import IsabelleCritique
import yaml
import argparse
import json
import os
import torch.cuda as cuda

def main(args):

    with open(f'data/{args.data_name}.json', 'r') as file:
        data = json.load(file)

    # Resolve output directory for checkpointing
    if args.run_id is None:
        mode = 'no_scheme' if args.no_scheme else 'scheme'
        prefix = f'{args.data_name}_run_'
        existing = [
            d for d in os.listdir('results')
            if d.startswith(prefix)
        ] if os.path.isdir('results') else []
        nums = []
        for d in existing:
            try:
                nums.append(int(d[len(prefix):].split('_')[0]))
            except ValueError:
                pass
        next_num = max(nums, default=0) + 1
        run_id = f'{prefix}{next_num:03d}_{mode}'
    else:
        run_id = args.run_id
    output_dir = os.path.join('results', run_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results directory: {output_dir}")

    # Save or load run config for reproducibility / resume safety
    config_path = os.path.join(output_dir, 'run_config.yaml')
    run_config = {
        'llm': args.llm,
        'data_name': args.data_name,
        'load_in_8bit': args.load_in_8bit,
        'no_scheme': args.no_scheme,
        'full_premise': args.full_premise,
        'max_iterations': args.max_iterations,
        'no_auto_assert': args.no_auto_assert,
    }
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        conflicts = {k: (saved_config[k], run_config[k])
                     for k in saved_config if saved_config.get(k) != run_config.get(k)}
        if conflicts:
            print(f"WARNING: CLI args differ from saved run config — overriding with saved values:")
            for k, (saved, current) in conflicts.items():
                print(f"  {k}: using {saved!r} (CLI had {current!r})")
            for k, (saved, _) in conflicts.items():
                setattr(args, k, saved)
    else:
        with open(config_path, 'w') as f:
            yaml.dump(run_config, f)
        print(f"Run config saved to {config_path}")

    # Initialise LLM once (lazy-loaded on first generate() call)
    model_id = args.llm
    device = 'cuda' if cuda.is_available() else 'mps'
    print(f"Using local model: {model_id} on device: {device}")
    llm = LocalLLM(model_id, device=device, load_in_8bit=args.load_in_8bit)

    seen_original_ids = set()
    for item in data:
        q_id = item['id']

        # With full-premise, all variants of the same original entry are identical — skip duplicates
        if args.full_premise:
            original_id = item.get('original_id', q_id)
            if original_id in seen_original_ids:
                print(f"Skipping {q_id}: duplicate original_id {original_id} under --full-premise")
                continue
            seen_original_ids.add(original_id)
        premise = item['premise']
        hypothesis = item['hypothesis']
        generated_premises = item['explanation']
        data_name = f'{args.data_name}_{q_id}'

        # Checkpoint: skip if result already exists
        result_path = os.path.join(output_dir, f'{data_name}_results.json')
        if os.path.exists(result_path):
            print(f"Skipping {data_name}: checkpoint exists")
            continue

        if premise == 'none':
            premise = None

        # Use all original premises (no removed premise) if requested
        if args.full_premise:
            all_orig = item.get('all_original_premises')
            if all_orig:
                premise = all_orig

        removed_premise = item.get('removed_premise')
        removed_premise_type = item.get('removed_premise_type')
        all_original_premises = item.get('all_original_premises')

        # Get argumentation scheme from data (suppressed if --no-scheme)
        argumentation_scheme = None if args.no_scheme else item.get('argumentation_scheme', None)
        if argumentation_scheme:
            print(f"Using argumentation scheme: {argumentation_scheme}")
        elif args.no_scheme:
            print("Scheme suppressed (--no-scheme)")

        isabelle_solver = IsabelleCritique(
            generative_model=llm,
            isabelle_session='HOL',
            theory_name=data_name,
            argumentation_scheme=argumentation_scheme
        )
        prompt_dict = {
            'refine axioms': 'refine_axioms_prompt.txt',
            'convert to nl': 'convert_to_nl_prompt.txt',
        }

        refinement_model = RefinementModel(
            generative_model=llm,
            critique_model=isabelle_solver,
            prompt_dict=prompt_dict,
            auto_assert=not args.no_auto_assert and not args.no_scheme,
        )

        # premise and generated_premises are optional
        # iterative refinement times are set to 10 by default
        results = refinement_model.refine(
            hypothesis=hypothesis,
            premise=premise,
            generated_premises=generated_premises,
            data_name=data_name,
            iterations=args.max_iterations,
        )
        item['results'] = results

        item['ground_truth_premise'] = removed_premise
        item['ground_truth_premise_type'] = removed_premise_type
        item['all_original_premises'] = all_original_premises
        if generated_premises:
            item['predicted_premises'] = generated_premises

        with open(result_path, 'w') as file:
            json.dump(item, file, indent=4)
        isabelle_solver.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', '-l', type=str)
    parser.add_argument('--data_name', '-d', type=str,
                        default='nlas-multi-preprocessed')
    parser.add_argument('--load_in_8bit', action='store_true',
                        help='Load model in 8-bit quantization (requires bitsandbytes, CUDA only)')
    parser.add_argument('--no-scheme', action='store_true',
                        help='Ignore argumentation_scheme field in data; run without scheme-guided formalisation')
    parser.add_argument('--full-premise', action='store_true',
                        help='Use all_original_premises instead of premise (no removed premise)')
    parser.add_argument('--max_iterations', '-i', type=int, default=8,
                        help='Maximum refinement iterations per example (default: 8)')
    parser.add_argument('--run_id', '-r', type=str, default=None,
                        help='Run identifier for checkpointing. Results saved to results/<run_id>/. '
                             'Reuse the same run_id to resume an interrupted run.')
    parser.add_argument('--no-auto-assert', action='store_true',
                        help='Disable auto-assertion of unsolved goals as axioms on stagnation')
    args = parser.parse_args()
    main(args)