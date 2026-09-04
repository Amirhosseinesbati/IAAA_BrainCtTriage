# A7 evaluation readiness and verified metric delivery

Previous goal turn: progress (implemented/tested A7 and started actual GPU
training). This turn revalidated supervisor parent47176/control child47321 as
live; child was using9900MiB on3090. No training parameter or source was changed.
This is a verified wait on specific live processes, not a stale lock inference.

## MLflow delivery verified, without using epoch metrics for model selection

Control run0f5b17c509714f7fa3da96726c59cfb6 had one completed epoch available
when delivery was checked. A server-side script compared every recorded value
for train_loss,supervised_loss,consistency_js,seconds,peak_vram_gib at the matching
step against MLflow history. All matched exactly. Only boolean delivery evidence
and the count were returned to the agent, not metric values.

Receipt locally preserved:
A7_FIRST_TRAINING_METRICS_RECEIPT_20260904.json
SHA256 eebbc8b9ac1e0064ee9667d6e5b48457e620d4179bd4c1e4dfbdda7f4aba3d75.
This supersedes the launch-time observation that no epoch metrics were yet
available; neither observation implies a quality result.

## Ready after BOTH arms finish, not executed during training

On the target server, from/workspace/IAAA_BrainCtTriage_mls_da:

```
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv/bin/python scripts/evaluate_mls_a7_pair.py
```

The wrapper refuses unfinished training or existing audit outputs. It rebuilds
all15-epoch/8115-step input-exposure/initialization checks from arm histories
instead of trusting the sequence's matched flag. It verifies checkpoint bytes,
embedded protocol/arm/epoch/initialization, equal configs and pinned source.
It then invokes the unchanged checksum-bound canonical CUDA evaluator once per
arm, sequentially. Both use the independent qualified same-runtime reference.
No CPU model execution, no extra training, no automatic replication/promotion.

The final comparison rechecks70-study scope, runtime/hardware/signature,
checkpoint identity, baseline/bounds, finite metric ranges and the objective
formula. Pass flags are recomputed from metrics. Even if consistency beats
the paired control, every frozen baseline resource gate is still mandatory.
If only the paired supervised control succeeds, do not credit consistency.
Full final triage/frozen-Champion validation is still required before release.

Seven new tests and seven existing runtime-reference tests passed on the server.
They cover mismatched exposure, invalid pass flags, runtime/coverage/checkpoint
changes, fabricated/nonfinite objectives, equal-model negative control and
nonpromotion even when the resource comparison succeeds. These are evaluator
tests, not completed evaluations of the still-training A7 models.

Wrapper SHA256:
4e3ac29a55733d076297b5981ee6034ace051be6f778257ff33bd9eead97b32c.
Test SHA256:
7d9f5e461a25e62076982b10c2adf957e56d396ab77f189541ffbba5facf506e.

No new experiment was added. Supervisor remains the finite control-then-
consistency sequence. The15-minute monitor is not active. Prefer waiting for
that sequence's completion over repeated epoch inspection. For any failed
observation, recheck the same process before interpreting it as termination.
If checkpoint creation completed but MLflow upload failed, reconcile/upload
the existing checkpoint instead of repeating training.

Goal remains active and unachieved. No claimed MLS/triage improvement, no local
best-model replacement, no submission ZIP.
