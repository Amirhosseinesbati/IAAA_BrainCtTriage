# Conservative three-fold OOF plan after Exp20 failure

## Frozen question

Exp20 failed both its primary epoch21 transfer and the one allowed named-best
diagnostic on held-out fold2. No additional Exp20 checkpoint, alpha, pooling
rule or threshold may be screened. The bounded next question is whether the
two independently passing regression components can improve the existing
three-member baseline without changing the failed fold2 member.

## Frozen recipe

- fold0: 90% Exp16 best-selector-AUC epoch16 plus 10% Exp19 epoch21 `mls_mm`;
- fold1: 90% Exp09 epoch15 plus 10% Exp18 epoch21 `mls_mm`;
- fold2: Exp15r epoch17 unchanged;
- selector, peak/ranking, heatmap and the locked `severity_window` pooling
  profile remain supplied by each baseline member;
- no model or image inference is permitted: inputs are completed CUDA-audit
  predictions;
- fold study IDs must be disjoint and every previously reported per-fold
  baseline/candidate metric must reproduce within absolute tolerance `1e-9`;
- only aggregate JSON/Markdown may leave Vast. Study-level prediction CSVs
  remain excluded.

The Exp18 choice is epoch21, not its epoch12 online best. Epoch21 was the fixed
full-study component-screen selection that generated the passing fold1 result.

## Frozen aggregate gates

Across the union of 70 fold0, 67 fold1 and 67 fold2 held-out studies:

1. candidate micro MAE must be no worse than baseline;
2. candidate micro Boundary-F1 must be no worse than baseline;
3. candidate micro selection objective must improve by at least `0.01`.

A deterministic 2,000-replicate paired study bootstrap with seed `20260902`
is diagnostic uncertainty evidence only; it cannot rescue a failed numerical
gate. Passing makes this a package candidate, not a leaderboard-proven release,
because actual test-time multi-model integration, CUDA parity, latency and the
official leaderboard remain separate gates.

## Pre-result parity correction

The first evaluator invocation stopped before creating any OOF output because
the spec used the historical Exp09 baseline MAE `1.2586648668`, while the
Exp18 component screen's own serialized grid reports `1.2590358221` and
objective `1.6115781950` for the same input SHA-256
`bb41e3d83e6eea2f6761af0aafbac88b1a0bf136267e4cdcc8900695ccee0d43`.
That screen had explicitly accepted the `0.0003709553` residual under its
`0.001` backward-parity tolerance. The OOF spec was corrected to reproduce the
actual serialized screen baseline at `1e-9`; candidate inputs, blend weights,
pooling, aggregate gates and bootstrap settings were not changed. This
correction occurred before any aggregate OOF result existed.
