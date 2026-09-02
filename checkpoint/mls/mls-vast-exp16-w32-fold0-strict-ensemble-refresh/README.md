# MLS fold-0 candidate: Exp16

- Selected file: `mls_multitask_best_selector_auc.pth`
- Training run: `mls-vast-exp16-w32-fold0-strict-ensemble-refresh`
- Selected epoch: 16 (highest online selector AUC)
- MLflow run ID: `a2478b8410d74de2b2806ef08d79051d`
- Training commit: `760b5648b027e24c3e88bfc413e0f4b01e95b898`
- Compute policy: strict deterministic CUDA-only; no model training or inference fallback to CPU
- SHA256: `bddcda5013cb88905a421095e71a28189181fde657aa3576be88f276d88ad15b`
- Size: 124,914,021 bytes

The checkpoint was selected only after independent full-series GPU inference on
all 70 fold-0 studies. Under the previously locked production pooling profile
(`severity_window`, radius 3, selector gate 0.5, minimum 3 active slices,
quantile 0.75, probability weighted, heatmap guard 0), it achieved:

- MAE: 1.6044777010 mm
- RMSE: 3.3514745762 mm
- Bias: +0.0375875063 mm
- Boundary F1: 0.8273325590
- Selection objective: 1.9498125829

It passes all preregistered fold-0 promotion gates and replaces the historical
fold-0 checkpoint as the current candidate. Keep the historical model until
the isolated submission package passes exact-runtime GPU validation and real
leaderboard validation.

See `reports/mls_experiments/mls-vast-exp16-w32-fold0-strict-ensemble-refresh/`
for the complete training trajectory, ten-checkpoint audit, raw full-series
predictions, pooling grid, promotion decision, and final report.
