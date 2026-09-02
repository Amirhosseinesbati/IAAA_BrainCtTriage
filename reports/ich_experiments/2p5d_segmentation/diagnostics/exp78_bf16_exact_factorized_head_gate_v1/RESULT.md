# Exp78 result — BF16-exact factorized composition passed

Exp78 completed on commit `a662053` using the real Exp61 checkpoint and CUDA
BF16 autocast, but no patient images, optimizer or held-out fold. All 91 related
tests passed. MLflow run: `edbd430578b94903831a27b8f9982a5a`.

The revised log-sum-exp-centered residual composition returned exactly the same
legacy logits, probabilities, hard argmax and auxiliary classification logits:
all four maximum differences were `0`. Both residual outputs were exactly zero.
Foreground→subtype and subtype→foreground cross-gradients remained only
`1.5768e-8` and `5.5879e-9`, respectively. Encoder/classifier trainable counts
were zero, the intended `2,837,996` spatial parameters and module modes were
confirmed, and all ten technical gates passed.

Decision: `authorize_preregistered_calibration_smoke`. This closes the BF16
identity defect found by Exp77 without relaxing its metric gate or changing its
loss/optimizer recipe.

Aggregate artifact SHA-256:
`05978f12e2899e405756d5080e38bfb51bd4952ed26faabbecfe4c4c6be565fd`.
