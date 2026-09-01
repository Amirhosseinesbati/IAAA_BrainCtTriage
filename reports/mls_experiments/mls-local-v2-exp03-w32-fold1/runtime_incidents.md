# Runtime incidents — mls-local-v2-exp03-w32-fold1

## 2026-08-27 — GPU power cap during epoch 7

- Epochs 1–6 trained CUDA-only at approximately 1.78–1.82 batches/s.
- At the start of epoch 7, throughput fell to approximately 0.37 batches/s
  (2.68–2.71 seconds/batch).
- `nvidia-smi` showed P5, 810 MHz memory clock, about 28–29 W power draw,
  `SW Power Cap: Active`, and `SW Thermal Slowdown: Active`.
- The current ceiling was 30 W while both the requested and default GPU
  power limits were 60 W. Hardware thermal and power-brake slowdown were not
  active; observed temperature fell from 61 C to 58 C.
- An approved attempt to restore the vendor-default 60 W limit with
  `nvidia-smi -pl 60` was rejected by the driver for insufficient operating
  system permissions. No device setting was changed.
- Windows reported only the Balanced power scheme. Training remained finite,
  CUDA-only, and checkpoint-safe. The best checkpoint at the onset of the
  incident was epoch 6 (selection objective 2.823, slice MLS MAE 2.210 mm,
  selector AUC 0.830, selector F1 0.779).
- During the latter part of epoch 7 the device recovered automatically to P0,
  6000 MHz memory clock and approximately 48 W. Because normal performance
  resumed without changing experiment state or hyperparameters, training was
  allowed to continue under the original early-stopping policy.
