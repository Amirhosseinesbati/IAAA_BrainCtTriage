# MLS experiment ledger

| UTC | Run | Status | Detail |
|---|---|---|---|
| 2026-08-26T21:00:00Z | mls-local-v2-exp01 | launcher_failed | ZenML global schema 0.96.2 is newer than project client 0.95.1; no training epoch started and no model output was produced. Direct validated launcher selected without modifying the global ZenML database. |
| 2026-08-27T00:32:37.488489+00:00 | mls-local-v2-exp01 | planned | multitask selector + spatial heatmap loss |
| 2026-08-27T02:39:06.912971+00:00 | mls-local-v2-exp01 | failed | UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f3c3' in position 0: character maps to <undefined> |
| 2026-08-27T02:43:11.368954+00:00 | mls-local-v2-exp01 | completed | Recovered MLflow teardown encoding error; existing run 8fee771402924977bcfdc6e028c6625e finalized. |
| 2026-08-27T03:40:55.184664+00:00 | mls-local-v2-exp02-w32 | planned | multitask selector + spatial heatmap loss |
| 2026-08-27T05:38:49.664224+00:00 | mls-local-v2-exp02-w32 | completed | best MLS MAE=1.8392, selector AUC=0.9130 |
| 2026-08-27T06:25:28.474352+00:00 | mls-local-v2-exp03-w32-fold1 | planned | multitask selector + spatial heatmap loss |
| 2026-08-27T08:57:33.712476+00:00 | mls-local-v2-exp03-w32-fold1 | completed | best MLS MAE=1.8772, selector AUC=0.9053 |
| 2026-08-27T09:09:21.230843+00:00 | mls-local-v2-exp04-w32-fold2 | planned | multitask selector + spatial heatmap loss |
| 2026-08-27T12:03:19.164683+00:00 | mls-local-v2-exp04-w32-fold2 | completed | best MLS MAE=2.2273, selector AUC=0.8985 |
| 2026-08-27T12:31:00.822289+00:00 | mls-local-v2-exp05-w32-fold0-peakaware | planned | multitask selector + spatial heatmap loss |
| 2026-08-27T15:15:05.981403+00:00 | mls-local-v2-exp05-w32-fold0-peakaware | completed | best MLS MAE=2.1999, selector AUC=0.9016 |
| 2026-08-27T15:36:03.642808+00:00 | mls-local-v2-exp06-w32-fold1-peakaware-transfer | planned | multitask selector + spatial heatmap loss |
| 2026-08-27T18:06:07.005619+00:00 | mls-local-v2-exp06-w32-fold1-peakaware-transfer | completed | best MLS MAE=1.7408, selector AUC=0.8945 |
| 2026-08-27T18:17:36.175598+00:00 | mls-local-v2-exp07-w32-fold2-peakaware-crossfold | planned | multitask selector + spatial heatmap loss |
| 2026-08-27T21:15:30.342916+00:00 | mls-local-v2-exp07-w32-fold2-peakaware-crossfold | completed | best MLS MAE=2.4219, selector AUC=0.9214 |
| 2026-08-27T21:55:09.871549+00:00 | mls-local-v2-exp08-w32-fold0-hybridsoft-snapshots | planned | multitask selector + spatial heatmap loss |
| 2026-08-28T01:04:50.484045+00:00 | mls-local-v2-exp08-w32-fold0-hybridsoft-snapshots | completed | best MLS MAE=2.1528, selector AUC=0.9082 |
| 2026-08-28T01:56:26.821092+00:00 | mls-local-v2-exp09-w32-fold1-hybridsoft-transfer | planned | multitask selector + spatial heatmap loss |
| 2026-08-28T04:32:48.797773+00:00 | mls-local-v2-exp09-w32-fold1-hybridsoft-transfer | completed | best MLS MAE=1.8134, selector AUC=0.9035 |
| 2026-08-28T05:02:11.069328+00:00 | mls-local-v2-exp10-w32-fold2-hybridsoft-transfer | planned | multitask selector + spatial heatmap loss |
| 2026-08-28T07:34:17.980552+00:00 | mls-local-v2-exp10-w32-fold2-hybridsoft-transfer | completed | best MLS MAE=2.3776, selector AUC=0.9143 |
| 2026-08-28T08:38:03.892176+00:00 | mls-local-v2-exp12-w32-fold2-studybalanced | planned | multitask selector + spatial heatmap loss |
| 2026-08-28T08:38:13.175713+00:00 | mls-local-v2-exp12-w32-fold2-studybalanced | failed | AttributeError: 'MLSHeatmapConfig' object has no attribute 'sampling_mode' |
| 2026-08-28T08:40:11.561422+00:00 | mls-local-v2-exp12r1-w32-fold2-studybalanced | planned | multitask selector + spatial heatmap loss |
| 2026-08-28T11:30:19.334583+00:00 | mls-local-v2-exp12r1-w32-fold2-studybalanced | completed | best MLS MAE=2.3615, selector AUC=0.9125 |
| 2026-08-28T11:56:47.979729+00:00 | mls-local-v2-exp13-w32-fold2-hybridsampler | planned | multitask selector + spatial heatmap loss |
| 2026-08-28T14:49:44.808470+00:00 | mls-local-v2-exp13-w32-fold2-hybridsampler | failed | MlflowException: API request to https://dagshub.com/amiresbati62/BrainCtTriage.mlflow/api/2.0/mlflow/runs/log-batch failed with exception HTTPSConnectionPool(host='dagshub.com', port=443): Max retries exceeded with url: /amiresbati62/BrainCtTriage.mlflow/api/2.0/mlflow/runs/log-batch (Caused by NameResolutionError("HTTPSConnection(host='dagshub.com', port=443): Failed to resolve 'dagshub.com' ([Errno 11001] getaddrinfo failed)")) |
| 2026-09-02T21:30:01.252919+00:00 | mls-vast-da-baseline-fold0-seed3407 | planned | multitask selector + spatial heatmap loss |
| 2026-09-02T22:56:00.481681+00:00 | mls-vast-da-baseline-fold0-seed3407 | completed | best MLS MAE=2.2463, selector AUC=0.9046 |
| 2026-09-05T14:41:22+00:00 | mls-reflection-control-fold1-seed42 | completed | R1 control، epoch15 ثابت، CUDA-only، MLflow=fbc9a8f38a74457eb45e9d48f60c31e8، checkpoint SHA=8a8240e0…e01102؛ هنوز promotion مجاز نیست. |
| 2026-09-05T15:17:04+00:00 | mls-reflection-control-fold1-seed42 | evaluated | strict raw-DICOM CUDA، 67/67؛ MAE=1.531936، F1@3=0.754717، F1@5=0.702703، Boundary-F1=0.728710؛ promotion_eligible=false. |
| 2026-09-05T15:22:36+00:00 | mls-reflection-control-fold1-seed42 | package_parity_passed | source-vs-submission CUDA، 67/67، batch=8، atol=1e-6، errors=[]، همهٔ max deltaها=0.0؛ ZIP هنوز مجاز نیست. |
| 2026-09-05T15:25:00+00:00 | mls-reflection-reflect-fold1-seed42 | started | candidate از پیش‌ثبت‌شدهٔ R1 شروع شد؛ تنها اختلاف با control، horizontal_flip_prob=0.5 است. |
| 2026-09-05T16:12:35+00:00 | mls-reflection-reflect-fold1-seed42 | evaluated | strict raw-DICOM CUDA، 67/67؛ MAE=1.224877، F1@3=0.842105، F1@5=0.777778، Boundary-F1=0.809942؛ نسبت به C0 در همهٔ gateهای MLS بهتر است، اما هنوز تک-seed است. |
| 2026-09-05T16:16:30+00:00 | mls-reflection-reflect-fold1-seed42 | package_parity_passed | source-vs-submission CUDA، 67/67، batch=8، atol=1e-6، errors=[]، همهٔ max deltaها=0.0؛ ZIP هنوز مجاز نیست. |
| 2026-09-05T16:20:08+00:00 | mls_reflection_r1_paired_screen_gate | passed | C1 در MAE، F1@3، F1@5 و Boundary-F1 نسبت به C0 non-inferior است؛ فقط R1R three-seed continuation مجاز شد، نه promotion. |
| 2026-09-05T16:55:06+00:00 | mls_reflection_r1r_replication | preregistered | contract SHA=1a5ac4ab…95a1؛ C0/C1 × seedهای 2026/3407 با epoch15 و recipe ثابت قفل شدند؛ promotion=false. |
| 2026-09-05T16:56:34+00:00 | mls_reflection_r1r_replication | static_validation_passed | parent R1، cache/truth/fold/frozen Champion، six-member config family و exact intervention hash-bound و passed شدند. |
