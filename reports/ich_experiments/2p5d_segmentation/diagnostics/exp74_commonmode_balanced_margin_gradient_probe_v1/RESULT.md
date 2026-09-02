# Exp74 result — common-mode isolation works, subtype objective still insufficient

Exp74 completed 24 train-only batches on commit `8a570ca`, with no optimizer or
held-out access. All 86 related tests passed. MLflow run:
`7e45b70c0d174c18b13f6a31b7cc7eb2`.

The common-mode derivative removed the main absolute IPH amplification: its
target-channel ratio changed from Exp73's `2.541x` to `0.988x`. Structural gates
remained safe (`background=0.614x`, decoder/head cosine=`0.435`). On the correct
margin metric, EDH passed at `1.299x`, but SAH=`1.022x`, IVH=`0.666x`,
IPH=`0.701x`, and especially SDH=`0.114x` failed.

Decision: `reject_exact_loss_weighting_before_calibration_or_outer`. Common-mode
foreground backpropagation is retained as a valid decoupling mechanism. The
remaining failure is in the subtype objective: conditional Balanced Softmax alone
does not retain the class-wise overlap/morphology pressure supplied by multiclass
Dice, especially for diffuse SDH/SAH. The next candidate must add conditional
subtype Dice and hard-example focal modulation without restoring background-
subtype coupling.

Aggregate artifact SHA-256:
`d07816885873458f12ff7468623bc2e31e26d1bd8c5dbe3d018953828991a3c0`.
