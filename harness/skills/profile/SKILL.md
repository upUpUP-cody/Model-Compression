---
name: harness-profile
description: >-
  Runs a cheap compressibility profile stub for Compression Harness before searching
  recipes. Use when the user asks for layer sensitivity probes, model profile, or
  profile_model primitive.
---

# Harness Profile

## Instructions

```bash
PYTHONPATH=harness/src python -m compression_harness.cli profile \
  --model /mnt/data/models/Qwen2.5-3B
```

v0.1 returns a dry-run profile placeholder. Persist real profiles under `harness/knowledge/profiles/` in Phase C.

## Examples

- "先给模型做 compressibility profile" → profile skill
