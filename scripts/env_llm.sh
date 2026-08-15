#!/usr/bin/env bash
# Phase K LLM environment: all large HF/model/dataset artifacts on /mnt/data.
# Usage: source scripts/env_llm.sh

export HF_HOME=/mnt/data/hf
export TRANSFORMERS_CACHE=/mnt/data/hf
export HF_DATASETS_CACHE=/mnt/data/datasets
export HF_HUB_CACHE=/mnt/data/hf/hub
# Mainland mirror; override to https://huggingface.co if needed.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export LLM_DATA_ROOT=/mnt/data
export LLM_MODEL_DIR=/mnt/data/models/Qwen2.5-1.5B-Instruct
export LLM_SQUAD_DIR=/mnt/data/datasets/squad
export LLM_RESULTS_ROOT=/mnt/data/results

mkdir -p \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$LLM_DATA_ROOT/models" \
  "$LLM_DATA_ROOT/results" \
  "$LLM_SQUAD_DIR"
