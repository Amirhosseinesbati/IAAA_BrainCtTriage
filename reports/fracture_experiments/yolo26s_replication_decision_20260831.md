# YOLO26s fracture replication decision — 2026-08-31

## Decision

Do not expand YOLO26s training to folds 0, 1, and 3 at this stage. The
architecture improved the small fold-4 development result, but failed the
pre-planned fold-2 replication against YOLOv8s. A rank blend selected only on
fold 4 also failed to improve the fixed YOLOv8s reference on fold 2.

This decision avoids spending three additional GPU training runs on a model
family that has not demonstrated cross-fold gain. It does not claim that
YOLO26s is universally inferior; the confidence interval remains wide because
fold 2 contains only eight positive studies.

## Reproducibility

- Vast.ai instance: `49251973` (RTX 3060 12 GB)
- Git branch: `codex/competition-winning-pipeline`
- Training run name: `fracture-v2-posr4-f2-yolo26s-lr5e4`
- MLflow run: `d45365f178aa49298c5a5c594ede5bc6`
- MLflow status: `FINISHED`
- Training data: `Data/processed/fracture_v2_balanced_r4/fold_2`
- Initialization: official `yolo26s.pt`
- Image size: 512
- Batch size: 16
- Learning rate: 0.0005
- Maximum epochs: 40
- Early-stopping patience: 15
- Actual last saved periodic checkpoint: epoch 30
- Bootstrap seed: 20260831
- Bootstrap iterations: 20,000

The post-training checkpoint summaries and comparison artifacts were attached
to the same MLflow run. Large model artifacts remain intentionally deferred.

## Fold-2 checkpoint screening

Study-level screening evaluated every saved checkpoint using max, top-2,
top-3, top-5, adjacent-pair, window-3, and noisy-or pooling. The best result
was:

- checkpoint: `epoch30.pt` (equivalent to `last.pt`)
- pooling: noisy-or
- AUC: **0.8961864407**
- validation studies: 67
- positives: 8
- negatives: 59

The Ultralytics detection-selected `best.pt` corresponded to the epoch-15
state and reached only 0.8622881356 with its best study pooling. This confirms
that detection mAP is not a safe checkpoint-selection proxy for the
study-level competition target.

Full screening summary:
`reports/fracture_experiments/yolo26s_f2_checkpoint_screen_summary.json`.

## Paired replication against YOLOv8s

Reference:

- YOLOv8s fold-2 epoch 10
- adjacent-pair pooling
- AUC: **0.9364406780**

Candidate:

- YOLO26s fold-2 epoch 30
- noisy-or pooling
- AUC: **0.8961864407**

Paired stratified bootstrap result:

- observed delta, candidate minus reference: **-0.0402542373**
- 95% bootstrap interval: **[-0.1207627119, 0.0233050847]**
- probability candidate is not better: **0.8839**

The interval crosses zero, so this is not proof of universal inferiority.
Nevertheless, the observed effect, its direction, and the high probability of
no improvement are insufficient to justify three more YOLO26s training runs.

Artifact:
`reports/fracture_experiments/comparisons/yolo26s_epoch30_noisyor_vs_yolov8s_epoch10_adjacent_f2.json`.

## Leakage-controlled rank-blend check

To test complementarity without tuning on the confirmation fold:

1. Both detectors were fixed at epoch 10.
2. YOLOv8s used noisy-or and YOLO26s used top-5 pooling on both folds.
3. Scores were converted to within-fold average percentile ranks.
4. Candidate weights from 0 to 1 in steps of 0.05 were searched on fold 4.
5. The selected weight was applied unchanged to fold 2.

Development fold 4:

- YOLOv8s: 0.8173302108
- YOLO26s: 0.8454332553
- selected candidate weight: 0.65
- selected blend: 0.8548009368

Confirmation fold 2:

- fixed YOLOv8s reference: **0.9237288136**
- fixed YOLO26s candidate: **0.8707627119**
- fixed selected blend: **0.9131355932**
- delta versus reference: **-0.0105932203**
- 95% bootstrap interval: **[-0.0847457627, 0.0529661017]**
- probability blend is not better: **0.62245**

The fold-4 gain did not replicate. The blend is therefore rejected rather than
added to inference complexity.

Artifact:
`reports/fracture_experiments/comparisons/yolo26s_yolov8s_fixed_epoch10_rank_blend_f4_to_f2.json`.

## Consequence for the next experiment

Keep the five-fold YOLOv8s snapshot ensemble as the current detector baseline.
The next justified model-family experiment is the already researched frozen-
feature smooth-attention MIL contingency. It must use patient-disjoint nested
validation, a small capacity head, explicit slice-order smoothness, and a
comparison against the unchanged YOLOv8s study scores. Naive neighbour-channel
2.5D stacking and further same-fold aggregator fitting remain rejected.

No YOLO26s fold-2 checkpoint is copied to `checkpoint/ich`, because it does not
meet the current acceptable-model gate.
