---
name: harness-recover
description: >-
  Compression Harness recover skill contract for LoRA / recalibration / distill when
  a run is slightly outside the near-lossless band. Use when the user asks to repair
  or recover a compressed model. Phase D only — do not pretend training ran.
---

# Harness Recover

## Instructions

1. **Contract only until Phase D.** Call CLI to confirm:

```bash
PYTHONPATH=harness/src python -m compression_harness.cli recover
```

2. Tell the user recover is not implemented; options now: milder recipe (`compress` + `plan-next`) or wait for Phase D.
3. Never start unauthorized finetune jobs or claim fake recovery metrics.

## Examples

- "用 LoRA 救一下精度" → explain Phase D; suggest milder recipe instead
