# Exp86 — independent SAH expert separability probe

## Rationale fixed before execution

Exp67 showed that class-safe background/IPH-to-SAH correction can improve SAH
Dice while leaving IVH/SDH/EDH exactly unchanged, but its SAH-volume gain was too
small. Exp68–71 showed that frozen incumbent features do not select errors with
enough precision. Exp72–75 closed loss-only changes to the shared six-class
softmax head, and Exp80–85 showed that factorized residual heads either perturb
shared morphology or trade SAH against SDH.

Exp86 therefore changes the learned representation: it copies the Exp61 decoder
into an independently trainable SAH decoder and attaches one binary head. The
incumbent encoder, decoder, six-class head and classification head remain frozen.
This removes six-way subtype competition from expert training without changing
the deployable incumbent in this diagnostic.

This design is consistent with published ICH systems that separate detection
and subtype/segmentation stages or use independently supervised outputs:

- Cho et al., *Improving Sensitivity on Identification and Delineation of
  Intracranial Hemorrhage Lesion Using Cascaded Deep Learning Models*:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6499861/
- Monteiro et al., *Accurate and Efficient Intracranial Hemorrhage Detection and
  Subtype Classification in 3D CT Scans with CNN/LSTM Networks*:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7582288/
- *Label-efficient deep semantic segmentation of intracranial hemorrhages in
  CT-scans*: https://pmc.ncbi.nlm.nih.gov/articles/PMC10406224/

## Locked execution

- Exp61 checkpoint and Schema4 manifest hashes are recorded before execution.
- Patient-safe outer fold `2` is never inferred. Calibration fold is `1`; seed
  is `42`.
- Exactly three epochs, batch `8`, AdamW `1e-4`, weight decay `1e-4`, cosine
  schedule and BF16 on the RTX 3090.
- Only the copied decoder and binary SAH head are trainable. The head starts from
  the incumbent SAH channel minus the mean background/IPH channel.
- Supervision is restricted to known pixels where the frozen incumbent predicts
  background or IPH; target is true SAH. Loss is `0.60` focal BCE with positive
  weight `16` plus `0.40` Tversky (`alpha=0.30`, `beta=0.70`).
- The calibration fold is evaluated once after the fixed final epoch. No epoch
  selection or threshold search is allowed.
- Only 4096-bin aggregate histograms are persisted. No slice/study/pixel-level
  predictions, outer inference, MLflow/DagsHub upload, Telegram message or GitHub
  push is performed.

## Locked gate

The primary score is expert probability multiplied by the frozen auxiliary SAH
slice probability, within incumbent background/IPH pixels. It is compared with
the analogous incumbent SAH-vs-background/IPH margin. Every condition is
conjunctive:

1. all required metrics are finite and at least 512 recoverable SAH pixels exist;
2. pooled average-precision gain is at least `0.005` and ratio at least `1.25x`;
3. pooled ROC-AUC gain is at least `0.01`;
4. within 15-pixel dilation of incumbent foreground, AP gain is at least `0.01`;
5. precision at 10% recall is at least `max(2%, 20 × prevalence)`.

Passing authorizes only a separately preregistered safe-fusion calibration
screen. Failure closes this exact independent SAH expert recipe. No checkpoint
is written unless every gate passes.
