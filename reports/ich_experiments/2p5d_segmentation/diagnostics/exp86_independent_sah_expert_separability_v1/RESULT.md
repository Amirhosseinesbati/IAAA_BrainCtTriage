# Exp86 result — independent SAH expert

Decision: `reject_independent_sah_expert_before_fusion_or_outer`.

The frozen Exp61 incumbent was paired with a trainable copied decoder and binary
SAH head for three BF16 epochs. The expert did not preserve useful ranking on
calibration1. Across incumbent-background/IPH pixels, expert raw AP/AUC were
`0.003873/0.51112`, below incumbent raw `0.031781/0.61461`. Near incumbent
foreground, expert raw AP was `0.027868`, below `0.061699`; on IPH-only pixels it
was `0.058048`, below `0.089685`. The positive prevalence in the broad region was
only `0.02228%`, so accuracy-like impressions would have been misleading.

The preregistered separability gate failed. Outer2 was not inferred, no row-level
predictions were persisted, no checkpoint was saved and no fusion was attempted.
This result rejects this exact full-decoder binary-expert design, not the general
idea that SAH may benefit from different representation or temporal context.

Runtime was `187.94s`, peak VRAM `1.270 GiB`, and the frozen incumbent checkpoint
SHA256 was `5f304018340e88e1d858d0842e432d8a163c96ed26117796c7068b6d399c18d4`.
The complete aggregate metrics are in `aggregate_result.json`.
