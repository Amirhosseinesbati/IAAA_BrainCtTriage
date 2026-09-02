# Exp15r strict MLS control: final audit and promotion record

## Outcome

Exp15r completed successfully and produced a new promoted fold-2 MLS candidate. The promoted model is the epoch-17 snapshot, selected by the preregistered historical production pooling profile rather than by a same-fold retuned rule.

- Training: 23/23 epochs, exit code 0, RTX 3060, CUDA-only, no CPU fallback.
- Determinism: strict PyTorch algorithms, deterministic cuDNN/cuBLAS policy, epoch-addressable RNG seeds and explicitly seeded DataLoader workers.
- MLflow run: `a35975ea4dd242f4b9c12dbdbc1e7491`.
- Training source commit: `8299c02b07c8ffa70d48e252baa112be2e9684fe`.
- Runtime: 2026-09-01 21:16:56 UTC to 22:43:25 UTC (about 86.5 minutes).
- Peak allocated training VRAM: 4.647 GiB.
- Data contract revalidated immediately before training: 3484 rows, 1781 positive rows, 1703 negative rows, 338 studies, 3484 unique resolved image paths, complete spacing and study truth.

## Infrastructure findings

The first strict smoke test correctly failed before model construction with CUDA error 803. `nvidia-smi` and device nodes were healthy, but the container loader preferred `/usr/local/cuda-12.6/compat/libcuda.so.560.35.05` over the host driver library `libcuda.so.580.82.09`. Putting `/usr/lib/x86_64-linux-gnu` first in `LD_LIBRARY_PATH` restored PyTorch CUDA 12.8 access to the RTX 3060. The persistent training wrapper already enforces this host-library priority. A strict HRNet-W32 512-pixel forward/backward smoke then completed with finite loss and about 0.996 GiB peak VRAM.

No model arithmetic was allowed on CPU. The failed smoke stopped explicitly at the CUDA gate and never fell back.

## Training trajectory

The deterministic control initially differed from the historical benchmark-mode trajectory but rapidly stabilized. Important validation milestones were:

| Epoch | Study MAE (mm) | Boundary F1 | Selector AUC | Selection objective | Interpretation |
| ---: | ---: | ---: | ---: | ---: | --- |
| 9 | 1.5439 | 0.8992 | 0.8784 | 1.8063 | first strong study-level checkpoint |
| 10 | 1.4136 | 0.8704 | 0.8924 | 1.7266 | early best MAE |
| 12 | 1.5751 | 0.9035 | 0.9141 | 1.8110 | strong boundary/selector balance |
| 16 | **1.3946** | 0.9124 | 0.9196 | **1.6099** | best internal objective/study checkpoint |
| 18 | 1.6845 | 0.9218 | **0.9202** | 1.8809 | best selector role |
| 22 | 1.4853 | **0.9361** | 0.9105 | 1.6580 | best boundary role |
| 23 | 1.4717 | 0.9035 | 0.9123 | 1.7085 | stable terminal checkpoint |

These training-time metrics only select candidates; they do not decide production promotion.

## GPU end-to-end audit

Ten candidates were evaluated on all 67 fold-2 studies: the six preregistered snapshots (13, 15, 17, 19, 21 and 23) plus best-objective, best-study, best-selector-AUC and best-study-boundary roles. All 670 candidate-study inference jobs completed on the RTX 3060 with zero failures. The resume-safe audit status records the checkpoint, start/end time and exit code for every candidate.

The same-fold diagnostic search found epoch 17 with a retuned severity-window profile at MAE 1.3886 mm, boundary F1 0.9131 and objective 1.5624. This is useful diagnostic evidence but is explicitly not used as the production estimate because checkpoint and pooling rule were inspected on the same fold.

## Locked-profile promotion result

The production comparison uses the historical cross-experiment locked profile: severity window size 3, selector gate 0.5, at least 3 active slices, quantile 0.75, probability weighting enabled and heatmap guard disabled.

| Candidate | Locked MAE (mm) | Locked boundary F1 | Objective | Decision |
| --- | ---: | ---: | ---: | --- |
| **Exp15r epoch 17** | **1.5484** | 0.8926 | **1.7632** | promoted primary |
| Exp15r best-boundary (epoch 22) | 1.6101 | **0.9131** | 1.7839 | retained alternate |
| Exp15r best-selector (epoch 18) | 1.6628 | 0.8885 | 1.8858 | retained diagnostic |
| Exp15r best-objective/study (epoch 16) | 1.7687 | 0.8803 | 2.0082 | not promoted under locked rule |
| Historical Exp10 epoch 15 | 1.7144 | 0.8947 | 1.9250 | previous trusted reference |

Against Exp10, epoch 17 improves MAE by 0.1660 mm (about 9.68%) and the locked selection objective by 0.1618, while boundary F1 changes by only -0.0021. It passes the preregistered promotion thresholds of MAE <= 1.75 mm and boundary F1 >= 0.88.

This result also resolves the Exp14r2 calibration concern: a strict training trajectory transfers under the frozen old pooling rule, whereas Exp14r2 epoch 16 had locked MAE 1.9681 mm and boundary F1 0.8186. The deterministic intervention materially improved calibration stability rather than merely producing an optimistic same-fold rule.

## Model handoff

The promoted checkpoint is `mls_multitask_epoch_017.pth` (124,898,469 bytes, SHA256 `e4c5f91c4e9fb97b766477615f6e42244bed2ee53f85c98f4f1353146cb6e16e`). The local copy was resumed after an interrupted SCP transfer and was accepted only after its size and SHA256 exactly matched the server file.

Exp10 should remain available as a rollback candidate until the new checkpoint is integrated into the full submission package and a real leaderboard result is obtained. The Vast instance remains running and was not stopped or destroyed.

## Next experiment decision

Do not launch the old sigma-annealing Exp15 manifest; its prerequisite Exp14r2 reproduction gate failed. The next useful work is integration/leaderboard validation of epoch 17 and, only if needed, a single controlled successor built on the strict baseline. The best-boundary epoch-22 checkpoint is a sensible complementary candidate for a later cross-validated ensemble, but no ensemble is promoted from same-fold evidence alone.
