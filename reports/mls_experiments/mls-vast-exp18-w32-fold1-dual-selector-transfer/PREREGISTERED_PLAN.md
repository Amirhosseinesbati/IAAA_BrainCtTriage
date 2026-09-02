# Exp18 strict fold-1 dual-selector transfer

## Evidence that triggers this experiment

Exp17 completed 23/23 epochs and its 11-checkpoint, 737-study CUDA audit with
zero failures. It nevertheless failed the frozen production profile:

- Exp09 fold1 epoch15 reference: MAE 1.258665, Boundary-F1 0.823729,
  objective 1.611207.
- Exp17 best locked candidate: MAE 1.399901, Boundary-F1 0.793443,
  objective 1.813016.
- A profile selected without fold1, using only fold0 and fold2, transferred to
  Exp09 with MAE 1.484237 but to Exp17 with MAE 1.817143 and Boundary-F1
  0.698396.
- A 90% Exp09 / 10% Exp17 slice-prediction blend did not help the locked rule:
  MAE changed from 1.258665 to 1.264027 and Boundary-F1 from 0.823729 to
  0.805556.

The negative blend result rejects another same-recipe seed as the first choice.
Exp17 improved target-vs-nontarget separation but did not transfer its severity
ranking/calibration. The existing one-logit selector is asked both to gate
target presence and to rank within-study severity, although these objectives
have different calibration requirements.

## Single conceptual change

Replace the single peak-aware selector logit with two logits from the same
lightweight head:

1. presence logit: binary target-vs-nontarget BCE, used only for the absolute
   study gate and minimum-active-slice count;
2. peak logit: soft target `slice MLS / official study maximum MLS`, used for
   top-slice ranking, component/window anchoring and probability weighting.

The two BCE losses are equally weighted and divided by two, keeping the total
selector-loss scale equal to the historical recipe. HRNet-W32, image channels,
heatmap/keypoint objectives, sampler, optimizer, LR, augmentation, seed,
23-epoch schedule and strict determinism are frozen.

## Safety and evaluation protocol

- Training, validation and checkpoint inference are CUDA-only; CPU model
  forward/backward fallback is forbidden.
- The immutable dataset contract remains 3484 rows, 338 studies, 1781 target
  and 1703 nontarget slices, with 67 held-out fold1 studies.
- Preserve named checkpoints, best peak-AUC, and epochs 13/15/17/19/21/23.
- After training, audit every candidate on all 67 studies using CUDA.
- Apply the already frozen production shape: severity-window radius 3,
  presence gate 0.5, minimum three active slices, peak-weighted q0.75 and no
  heatmap guard. Any fold1-fitted grid is diagnostic only.
- Do not stop or destroy Vast instance 49527185.

## Promotion gates

A checkpoint may replace Exp09 fold1 epoch15 only if all three hold under the
frozen dual-head interpretation above:

- MAE <= 1.258665 mm;
- Boundary-F1 >= 0.82;
- objective <= 1.611207.

If no checkpoint passes, retain Exp09 and use the result to decide whether the
next intervention is 2.5D context or regression-specific fine-tuning. Online
validation alone cannot promote a checkpoint.
