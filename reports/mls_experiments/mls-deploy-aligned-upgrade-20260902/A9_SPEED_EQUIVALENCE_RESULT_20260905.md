# A9 trainer speed-equivalence result

This is operational evidence, not an MLS candidate model and not a leaderboard result. It writes no checkpoint, consumes no validation images, and uploads no private predictions.

The benchmark repeated one exact epoch twice in each order (reference → optimized and optimized → reference), with seed 42, fold 0, batch 16, 169 optimizer steps, strict FP32, AMP off, and TF32 off. The only tested changes were zero-copy hashing of the historical image/coordinate input digest and moving loss-scalar transfer from each optimizer step to the end of the epoch.

All equivalence gates passed: the historical input digest, loss trace, refiner state, optimizer state, scheduler state, RNG state, frozen baseline, and repeated runs were identical. Median loop time improved from 40.7526 seconds to 32.1039 seconds: a 1.2694× speed-up.

Decision: adopt these two implementation details for future MLS candidate trainers only. Do not modify the pinned, rejected A9 checkpoint or interpret this as a model-quality gain. The metrics-only MLflow run is `d616931d222f4e14a9a9c471804ca2e4`, readback-verified after a non-GPU repair.
