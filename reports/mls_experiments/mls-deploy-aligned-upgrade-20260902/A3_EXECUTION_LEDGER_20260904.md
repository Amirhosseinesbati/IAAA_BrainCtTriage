# A3 execution ledger — fold 0 / seed 42

## Scope and current state

This is an execution ledger, not a validation result.  It contains no raw
medical data, per-study prediction, training log, or intermediate epoch metric.

- Run: `mls-vast-da-a3-study-bag-fold0-seed42`.
- Server launcher session: `mls_da_a3_f0_s42`.
- Durable launcher state observed at the 30-minute milestone: `running`.
- Start: `2026-09-04T04:25:06.570522+00:00`.
- Launcher exit code and finish timestamp remain null while the run is active.
- Compute policy reported by the tmux launcher: `cuda_only`; the frozen manifest
  additionally declares `cuda_only_no_cpu_fallback`.
- The only observed runtime evidence is process/GPU liveness; it is not used to
  select an epoch or to infer model quality.

## Provenance and frozen intervention

- Server Git base: `b321de5d11f4fd2b00b80eeef25eb56e068a015c`.
- Local A3 source implementation: commit `be3056e`; the direct server sync was
  checksum-verified before the CUDA preflight because noninteractive GitHub
  credentials are unavailable.
- A3 is the study-bag selection auxiliary intervention.  It preserves the
  original slice-class-balanced primary loader, absolute geometric MLS target,
  fold (0), seed (42), 23-epoch schedule, and fixed epoch-15 selection point.
- A2 signed geometry remains rejected and cannot be resumed or used as a rescue
  comparison.

## Capacity and operational controls

- The largest fold-0 positive-study bag contains 23 annotated target slices.
- CUDA-only forward/backward preflight passed at peak allocated VRAM
  `19.080746 GiB` on the RTX 3090.  Its server-only report checksum is
  `9623f00306fc245f352eb9d5bcebd7cf0511a23473d7ab360a4b46ea9be181d9`.
- Therefore primary slice batch size is intentionally frozen at 5.  Raising it
  would consume the headroom needed for the worst-case auxiliary bag and add a
  second confounded training change.
- No run log, tmux capture, tqdm output, private raw prediction, or per-epoch
  validation value is inspected while the job is active.

## Terminal protocol (pre-registered)

1. Confirm the durable launcher completed with exit code 0 and no concurrent
   CUDA process.
2. Run only the already staged `run_vast_mls_a3_fold0_seed42_resource_screen.sh`:
   CUDA-only inference of epoch 15 on the fixed 70 held-out fold-0 studies with
   selector gate 0.5, top-3, p90 aggregation, and min-active 1.
3. Log only aggregate audit evidence to MLflow.  Keep raw predictions private
   and server-only.
4. Accept expansion only if all locked gates pass: MAE <= 1.470959 mm,
   F1@3 >= 0.819672, F1@5 >= 0.736842, boundary F1 >= 0.778257, and objective
   <= 1.904444.  A failure records `rejected_stop_a3_expansion`; no rescue
   checkpoint, pooling change, ensemble, or additional A3 seed may start.
5. A pass authorizes only the two predeclared fold-0 seed replications.  It is
   not a promotion or submission authorization.

## Audit and transfer tooling staged ahead of completion

- Fail-closed A3 resource-screen tooling: local commit `b6f50d6`; server-side
  evaluator, runner, and plan were checksum-matched after a backup of the prior
  plan.  Runner shell syntax and evaluator Python syntax passed on the server.
- Checksum transfer support for the tmux `status.json` launcher format: local
  commit `0483e33`, with six passing unit tests and a PowerShell syntax check.
  It will be used only for a model that later satisfies the applicable release
  gate; it does not transfer private predictions.
