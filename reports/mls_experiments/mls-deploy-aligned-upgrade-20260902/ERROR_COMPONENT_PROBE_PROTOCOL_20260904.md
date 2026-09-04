# Fixed training error-component diagnostic

Before inspecting results: compare baseline and A7 consistency only on the previously frozen 128 positive training slices. Source script SHA256 `0c12c83c5ddd943a7a7c9244987aa071737d92522b618190fd81726428933dba`; local/server hashes agree. Images and truth must reproduce the existing input exposure hash. Both models use batch8, IEEE inference, DARK decoder, eval/inference_mode, CUDA-only model forwards. All model state including buffers must be unchanged. Data/source/checkpoint pins are checked; no private row outputs are saved.

Measure perpendicular and parallel localization errors of each point in the true reference-line frame. Compare full MLS error with two diagnostic oracle substitutions: true reference endpoints plus predicted outer point; predicted endpoints plus true outer point. These substitutions are never valid inference candidates. Their error differences are not additive causal effects because component errors can cancel and MLS takes an absolute value.

Report fixed strata: all, true absolute angle <=10 degrees, >10 degrees; show sample counts. These are observational training-only associations, not causal pose evidence or generalization estimates. No statistical significance, independence, or leaderboard claim is authorized by this diagnostic.

Use the results to choose a bounded next architectural decision, not to launch an unrestricted search. Baseline numerical reproduction should be checked against the previous translation probe. A7 still fails held-out resource gates regardless of training decomposition.

Operational note: first SSH stdin command failed before launching Python because its filename had a trailing carriage return. Corrected direct SSH invocation is session 60402; do not restart it based on an observation timeout. No scheduled monitor was created.
