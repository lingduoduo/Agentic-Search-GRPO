# Refuse a checkpoint that would carry no weights

## Goal

Stop `train_loop` reporting a successful resume that restored nothing but a step
number. A trainer that cannot checkpoint must be refused, loudly, before the run
starts — not silently skipped.

## The problem

`train_loop`'s checkpoint helpers reached the trainer through
`getattr(trainer, hook, None)` and skipped the call when it was absent, while
still writing and reading their own step manifest:

```python
def save_checkpoint(trainer, path, step):
    trainer_save = getattr(trainer, "save_checkpoint", None)
    if trainer_save is not None:          # absent -> skipped
        trainer_save(path)
    (Path(path) / "trainer_state.json").write_text(json.dumps({"step": step}))

def load_checkpoint(trainer, path):
    trainer_load = getattr(trainer, "load_checkpoint", None)
    if trainer_load is not None:          # absent -> skipped
        trainer_load(path)
    return int(json.loads((Path(path) / "trainer_state.json").read_text())["step"])
```

Only two trainers in the repo implement those hooks:

| Trainer | `save_checkpoint` / `load_checkpoint` |
|---|---|
| `LLMGRPOTrainer` | yes |
| `SearchAgentGRPOTrainer` (subclass) | inherited |
| `GRPOTrainer` (bandit) | **no** |
| `DPOTrainer` | **no** |
| `SFTTrainer` | **no** |

For the three that do not, the failure is silent in both directions:

- **Saving** produced a directory containing `trainer_state.json` and nothing
  else — a thing that looks like a checkpoint, holds no weights, and cannot be
  resumed from.
- **Resuming** read the step back and returned it, so the loop started at step
  N against a *freshly initialised policy*. Training restarted from scratch
  while the metrics counted from step N, and nothing in the logs said so.

This is a wrong-answer bug, not an inconvenience: the run completes, the numbers
look plausible, and the model is not the one the history describes.

## The change

A named error, and two places that raise it.

```python
class CheckpointNotSupportedError(RuntimeError): ...

def _require_checkpoint_hook(trainer, hook):
    fn = getattr(trainer, hook, None)
    if not callable(fn):
        raise CheckpointNotSupportedError(...)
    return fn
```

`save_checkpoint` and `load_checkpoint` now resolve their hook through
`_require_checkpoint_hook`, so neither can no-op — including when called
directly, outside `train_loop`.

`train_loop` additionally checks the **save** hook up front, before the first
step:

```python
if config.ckpt_dir is not None and config.ckpt_every:
    _require_checkpoint_hook(trainer, "save_checkpoint")
```

Checking at configuration time rather than at the first checkpoint is the point:
a run that cannot save is better stopped immediately than after it has spent the
hours the checkpoint existed to protect. The load hook needs no separate up-front
check — `load_checkpoint` is called on the first line of the loop.

The error message names the trainer class and the missing hook, and says what to
do: implement it, or run without checkpointing.

## What does not change

`ckpt_dir` without `ckpt_every` writes `metrics.jsonl` and never checkpoints, so
it stays available to every trainer. Only the two paths that produce or consume
a checkpoint directory are gated.

The one caller in-tree, `examples/run_retriever_aware_grpo.py`, drives a
`SearchAgentGRPOTrainer`, which inherits both hooks. It is unaffected.

## Testing

Test-first. Both guards were watched failing with `DID NOT RAISE` before the
implementation existed, against an `UncheckpointableTrainer` double that mirrors
`SFTTrainer`/`DPOTrainer`/`GRPOTrainer`'s shape.

Four tests: a refused resume (asserting no step runs), a refused
periodic-checkpoint configuration (asserting the refusal precedes step 0), the
metrics-only run that must keep working, and a capable trainer that must still
resume exactly as before.

## Scope

This fixes the silent no-op. It does not add checkpointing to `SFTTrainer`,
`DPOTrainer` or `GRPOTrainer` — whether those need it is a separate question,
and answering it by guessing is how the no-op got here.

Also unchanged: `train_loop` catches a failing step and continues, so a
persistently broken trainer still burns `max_steps` and returns an empty
history. That is a real second defect, but it is a different one, and folding it
into a correctness fix would blur what this PR is for.
