# Exp72 result — exact weighting rejected before calibration

Exp72 ran 24 deterministic train-only batches (batch size 8) from the locked
Exp61 checkpoint. It performed no optimizer step, wrote no row-level prediction,
and did not read calibration, outer, test, or leaderboard data. All 82 related
server tests passed. MLflow run: `b94d6c8ae5b34fdebe2e26ffdcb78cdb`.

The hierarchical objective remained broadly aligned with the incumbent decoder/
head gradient (`cosine=0.56375`) and reduced mean foreground-channel gradient on
true background to `0.61444x`. It did not meet the rare-subtype gates: EDH target-
logit attraction was `0.71413x` rather than at least `1.10x`, and SAH was
`1.09739x` rather than at least `1.25x`. IPH was over-amplified to `2.47457x`;
IVH and SDH were reduced to `0.77235x` and `0.38803x` respectively.

Decision: `reject_exact_loss_weighting_before_calibration_or_outer`. The
hierarchical foreground/subtype concept remains viable because it improved the
background-pressure gate and retained positive representation-gradient alignment.
The specific conditional CE plus independent OVR mixture is not balanced enough:
it reallocates gradient toward incumbent subtype errors dominated by IPH rather
than reliably strengthening both tail targets. The next diagnostic must decompose
the conditional CE and OVR contributions and change the subtype objective itself;
post-hoc relaxation of these gates is forbidden.

Aggregate artifact SHA-256:
`438797bbc7acb29d01650f2189d3209094ab746517ec68aa537a13b609f43eda`.
