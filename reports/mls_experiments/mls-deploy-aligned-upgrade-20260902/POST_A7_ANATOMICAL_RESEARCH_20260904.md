# Post-A7: anatomical-context research and a bounded next decision

SciSpace search covered CT MLS localization, falx landmarks and multi-slice context. Search abstracts are not treated as verified implementation evidence. The primary CAR-Net paper was then read in full HTML: https://arxiv.org/html/2007.05393v1 .

## Primary-source finding

CAR-Net uses a 2D image, multiscale feature refinement and a separate learned rigid-pose rectifier. Its connectivity penalty constrains neighboring coordinates along a densely annotated midline within an image, not neighboring CT slices. The rectifier's training targets derive from anterior/posterior landmarks. Evaluation uses manually delineated slices and line-distance measures, not this competition's end-to-end triage metric. Therefore this paper does not establish that 2.5D input will improve our MLS model or that its full curve loss can be supervised by three isolated points. Its separately split datasets are not evidence of the cross-site transfer of one unchanged model.

## Relation to our evidence

Existing decoder-probe evidence identifies the third/outer-falx landmark as the largest mean localization error, but that Euclidean error includes displacement along the ideal midline, which need not cause MLS error. A7 reduces average translation sensitivity without passing held-out study gates. Neither observation yet isolates anatomical pose as the cause. Current HRNet already fuses multiscale features; calling it a shallow model or blindly replacing it with CAR-Net is unsupported.

No new architectural run is authorized by this research note alone. Avoid another loss-weight sweep on the repeatedly observed fold0.

## Next measurement before model design

Use only the fixed fold0 training population to quantify annotation geometry: anterior/posterior line angle, midpoint displacement, segment length, and outer-falx location projected parallel/perpendicular to that line. Return aggregate distributions only. Check finite coordinates, positive spacing, nondegenerate segments and exact dataset/fold hashes. No held-out labels/images or new model forward are needed for this metadata-only diagnostic on the server.

This can determine whether pose/scale variation is actually material and whether a landmark-relative representation is well-defined on our labels. It cannot prove a learned rectifier improves performance. Any later pose-conditioned refinement must use predicted geometry at deployment, avoid ground-truth ROI leakage, retain patient-level folds, and be evaluated through the complete frozen triage pipeline. Do not synthesize a dense anatomical curve by simply joining the three annotations and call it ground truth.

A2 already tested signed-offset supervision, so a relative-coordinate proposal must not repeat that loss under a new name. A3/A4/A5 already tested study-bag and ranking routes, and A6 local decoding. Their measured failures remain; no resource gate is relaxed here.
