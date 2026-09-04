# Same-runtime baseline qualified; negative control rejected as an upgrade

The previous goal turn made progress: two independent baseline executions
qualified an exact same-runtime reference. This continuation verified the
completed negative-control process (exit0), its70-study aggregate and hashes.

Independent executions A/B have identical private predictions including all
slice outputs, SHA653f3a5c591a3fd7b25181443d08bed663d5e49227f6b055678ab688d5842727.
Raw byte and ordered SOP fingerprints agree with the pinned anchor. Original
study decisions at1/3/5mm are unchanged. Historical cross-runtime parity still
FAILED and is not relabeled. Only the explicit same-runtime reference qualifies.

Baseline-as-candidate reproduced exactly the same private prediction SHA.
MAE1.4705651353mm, F1@3=.8196721311, F1@5=.7368421053,
Boundary-F1=.7782571182, objective1.9140508989. All noninferiority checks pass;
the required objective<=1.9040508989 fails. resource_gates_passed=false,
promotion_eligible=false, submission_zip_allowed=false. Correct behavior:
reproducing an existing model must not be called an upgrade. Runtime49.40s.

Locally preserved aggregate files and SHA256:

- same_runtime_baseline_qualification_20260904.json:
  59743c79a788839940f73e6d9e81cbd564c0fae9421239e2382debf3b56e5b19
- IEEE_BASELINE_INDEPENDENT_A_20260904.json:
  1ea5d24d340bc461cc489fdc1252e249e96bca2106ca2b4eeb961eb1ace93a8c
- IEEE_BASELINE_INDEPENDENT_B_20260904.json:
  d4e48f1f4beaa89fc574b1566fb291c526353021d13db8fc996f6b1f78eb0962
- CANONICAL_REFERENCE_NEGATIVE_CONTROL_20260904.json:
  d7410889dfef11301420a841e946ec0003bf7aee54d22b7c6b48680d6a049994

Server campaign:/workspace/iaaa_artifacts/mls_deploy_aligned_20260902.
Private prediction files stay there. No new winning checkpoint, no ZIP.
Future three-seed and cross-fold candidates need corresponding same-runtime
controls; the old median caches are not silently interchangeable.
