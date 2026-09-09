#!/usr/bin/env bash
# Phase K LLM environment: weights/data on /mnt/data; K5+ run artifacts on /mnt/data2.
# Usage: source scripts/env_llm.sh

export HF_HOME=/mnt/data/hf
export TRANSFORMERS_CACHE=/mnt/data/hf
export HF_DATASETS_CACHE=/mnt/data/datasets
export HF_HUB_CACHE=/mnt/data/hf/hub
# Mainland mirror; override to https://huggingface.co if needed.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ALLOW_CODE_EVAL=1
# Reduce CUDA allocator fragmentation during long lm-eval cascades (batch fallback).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export LLM_DATA_ROOT=/mnt/data
export LLM_MODEL_DIR=/mnt/data/models/Qwen2.5-3B-Instruct
export LLM_SQUAD_DIR=/mnt/data/datasets/squad
export LLM_GLUE_DIR=/mnt/data/datasets/glue

# K5+: large run outputs on the second volume.
export LLM_RESULTS_ROOT=/mnt/data2/results
export LLM_CHECKPOINT_ROOT=/mnt/data2/checkpoints
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/data2/hf/triton}"

# Compression Harness: multi-GB ckpt/runs + torch temp off root FS.
export HARNESS_EXPERIMENTS_DIR="${HARNESS_EXPERIMENTS_DIR:-/mnt/data2/results/harness_experiments}"
export TMPDIR="${TMPDIR:-/mnt/data2/tmp}"

mkdir -p \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$LLM_DATA_ROOT/models" \
  "$LLM_DATA_ROOT/results" \
  "$LLM_SQUAD_DIR" \
  "$LLM_GLUE_DIR" \
  "$LLM_RESULTS_ROOT" \
  "$LLM_CHECKPOINT_ROOT" \
  "$TRITON_CACHE_DIR" \
  "$HARNESS_EXPERIMENTS_DIR" \
  "$TMPDIR"
