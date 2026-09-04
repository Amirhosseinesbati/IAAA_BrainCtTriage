# A5 detached-rank preflight reconciliation

No A5 training, validation, raw-data inference, checkpoint evaluation, or
leaderboard submission occurred before this correction.

The initial synthetic W32/512/batch-10 preflight used `backbone.eval()` during
the rank-only forward to freeze BatchNorm buffers.  CUDA execution and memory
were valid (`8.296 GiB` peak), but its synthetic rank loss was `288.01`, versus
`0.6898` in the otherwise comparable A4 preflight.  The discrepancy identifies
a train/eval normalization-distribution mismatch in the draft isolation method;
it is not a performance result and must not be interpreted as an MLS metric.

Before training, A5 was revised to preserve the backbone's caller mode while
temporarily setting each BatchNorm module's `track_running_stats` to false
under `torch.no_grad()`.  This retains batch-statistic semantics but forbids
both backbone gradients and BatchNorm buffer updates.  The CUDA regression test
now asserts selector-only gradients, unchanged heatmap/backbone gradients,
unchanged running mean/variance, and restoration of the tracking flag.  A new,
distinct synthetic preflight artifact is required to pass before the first A5
training launch.
