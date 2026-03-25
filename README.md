# DRFT Benchmark

A benchmarking pipeline for evaluating multiple LLMs on the DRFT dataset using a multi-agent revision workflow and automatic quality metrics.

## Overview

This project runs the same DRFT task across a list of models and compares outputs using:

- **Multi-agent generation pipeline** (planner + writer)
- **Reference baseline scoring** against the gold dataset
- **Automatic metrics**, including BERTScore and G-EVAL-derived scores
- **Per-model CSV outputs** and pre-generated analysis charts

## Repository Structure

- `main.py` – main experiment entrypoint (model download + benchmark loop)
- `config.py` – runtime configuration, model list, metric flags, and dataset paths
- `agents/` – agent definitions and graph pipeline orchestration
- `services/` – baseline and LLM service/factory utilities
- `metrics/` – metric computation and result export helpers
- `data/` – DRFT input/output datasets
- `results/` – benchmark CSV outputs and generated graphs
- `comparison_results/` – qualitative per-role comparison text artifacts

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Before running, review and adjust `config.py` and/or environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `IP_LOCAL_SERVER`
- `DRFT_INPUT_CSV`
- `DRFT_OUTPUT_CSV`
- `DRFT_TOP_K`
- `DRFT_G_EVAL_ENABLED`
- `DRFT_BERT_BASELINE`

> Note: the current code sets a placeholder API key value in `config.py`. Replace it with your real key via environment variables for safer usage.

## Running the Benchmark

Run the full experiment loop:

```bash
python main.py
```

What this does:

1. Iterates through `MODELS_TO_COMPARE`
2. Pulls non-OpenAI models through the Ollama pull endpoint
3. Runs the DRFT multi-agent pipeline for each model
4. Saves per-model outputs to `results/results_<model>.csv`
5. Runs a gold-dataset baseline and logs comparative metrics

## Outputs

- **CSV metrics per model** in `results/`
- **Charts** in `results/graphs/`
- **Comparison text files** in `comparison_results/`

## Notes

- Default paths and server endpoints are currently tailored to the local environment in this repository.
- If you use a remote or custom inference server, update URL-related settings in `config.py`.
- For reproducibility, keep dataset versions fixed and record your environment variables when running comparisons.
