# Do Argumentation Schemes Help? An Assessment of Their Role in Formalising Natural Language Arguments

**Ameer Saadat-Yazdi, Victor David, Anthony Hunter, Serena Villata**  
Université Côte d'Azur, Inria, CNRS, I3S · University College London  
*Under review*

We investigate whether argument schemes, formalised as higher-order inference rules, can guide LLM-driven construction of Isabelle/HOL proofs from natural language arguments. Evaluating on 200 English arguments from the [NLAS dataset](https://doi.org/10.1016/j.dib.2024.111087) across 18 Walton scheme types, we compare a scheme-guided pipeline against a scheme-free baseline and analyse four key sources of failure.

## Overview

The pipeline translates natural language arguments into Isabelle/HOL proofs in three stages:

1. **Text-to-HOL** — each sentence is incrementally mapped to typed higher-order predicate calculus, sharing a predicate vocabulary with the formalised scheme
2. **Proof Template** — scheme metavariables are bound to concrete terms from the formalised argument; a structured ISAR proof template with `sorry` placeholders is generated
3. **Iterative Proof Completion** — Sledgehammer attempts to discharge each `sorry`; unresolved steps trigger bridge axiom generation (scheme gap) or claim-bridging premise generation (claim gap)

The baseline omits scheme formalisation, metavar binding, and proof-template generation, reproducing the pipeline of [Quan et al. (ACL 2025)](https://aclanthology.org/2025.acl-long.867/).

## Installation

### Python

```bash
pip install -r requirements.txt
```

> This project requires `isabelle-client==1.0.1`.

### Isabelle

This project uses **Isabelle2025-2**. Download it from [isabelle.in.tum.de](https://isabelle.in.tum.de/) and add the `bin` directory to your PATH.

**Linux:**

```bash
wget https://isabelle.in.tum.de/dist/Isabelle2025-2_linux.tar.gz
tar -xzf Isabelle2025-2_linux.tar.gz --no-same-owner
export PATH=$PATH:/path/to/Isabelle2025-2/bin
```

**macOS:**

```bash
wget https://isabelle.in.tum.de/dist/Isabelle2025-2_macos.tar.gz
tar -xzf Isabelle2025-2_macos.tar.gz --no-same-owner
export PATH=$PATH:/path/to/Isabelle2025-2.app/bin
```

**Jupyter:** Both Jupyter and `isabelle-client` use `asyncio`, so nested event loops must be enabled:

```python
import nest_asyncio
nest_asyncio.apply()
```

### Configuration

Set your API key and Isabelle master directory in `config.yaml`:

```yaml
isabelle:
  master_dir: '/absolute/path/to/formalising_arg_schemes/formalisation/isabelle'
```

## Usage

```bash
python main.py --llm <model_name> --data_name <dataset_name>
```

**Example** — run on the included example arguments:

```bash
python main.py -l gpt-4o -d example
```

**Reproducing paper results** — run on the NLAS-multi-preprocessed dataset with full premises and auto-assertion disabled:

```bash
python main.py -l <model_name> -d nlas-multi-preprocessed --full-premise --no-auto-assert
```

To run the baseline (no scheme guidance), add `--no-scheme`:

```bash
python main.py -l <model_name> -d nlas-multi-preprocessed --full-premise --no-auto-assert --no-scheme
```

Supported models: `gpt-4o`, `qwen`, `codestral`, `gemma`. See `generation/` for details and `config.yaml` for API key configuration.

## Results Summary

| Model | Scheme | Baseline |
| --- | --- | --- |
| Qwen2.5-Coder-32B | 35.5% | 24.0% |
| Gemma-4-31B | 10.5% | 8.5% |
| Codestral-22B | 1.5% | 0.5% |

When schemes are successfully instantiated, only 10% of attempts fail at the final claim step — confirming that schemes reliably close the last inferential step, with the bottleneck being scheme premise satisfaction. See the paper for full error analysis.

## Citation

```bibtex
@unpublished{saadatyazdi-etal-2026-argumentation,
    title  = "{D}o Argumentation Schemes Help? {A}n Assessment of Their Role in Formalising Natural Language Arguments",
    author = "Saadat-Yazdi, Ameer and David, Victor and Hunter, Anthony and Villata, Serena",
    note   = "Under review",
    year   = "2026",
}
```

This work builds on the LLM-theorem-prover pipeline of [Quan et al. (ACL 2025)](https://aclanthology.org/2025.acl-long.867/).
