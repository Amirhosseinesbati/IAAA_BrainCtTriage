# Fracture detector + SA-MIL candidate

Five YOLOv8s epoch-10 outer-fold detectors plus three tiny SA-MIL heads per fold. Slice scores and embeddings are mapped through train-only empirical CDFs, blended, averaged across folds, and decision-aligned so the official 0.5 cutoff corresponds to OOF score 0.836321.

Offline evidence: deployable OOF AUC 0.9078; leakage-controlled decision F1 0.5484 (precision 0.50, recall 0.607). This is a best-current candidate, not a leaderboard-certified final model.
