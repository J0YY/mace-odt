# Exact immediate-consumer overnight run

This launch is frozen before held-out evaluation. It uses the existing 64
configuration evaluation manifest and the disjoint 128 configuration discovery
manifest. It does not read DFT labels, fit an offset, or select a rank from the
evaluation results.

## Launch

First submit the exact environment job. Submit the fidelity job only after the
environment job succeeds.

```bash
environment_job=$(sbatch --parsable cluster/immediate_consumer_environment.sbatch)
sbatch --dependency="afterok:${environment_job}" cluster/immediate_consumer_fidelity.sbatch
```

The fidelity job requests eight CPU cores, 96 GiB of memory, and twelve hours.
The expected runtime is four to eight hours. The larger memory and time limits
leave room for all six ranks and fourteen methods on the 64 frozen structures.

The final artifact is
`results/mace_off23_small_immediate_consumer_fidelity_64.json`. The job refuses
to replace an existing result. The evaluator must validate the exact-environment
JSON and NPZ pair, including the embedded NPZ hash, before loading any held-out
geometry.

## Tomorrow's decision

The complete frozen protocol is in
`configs/immediate_consumer_fidelity_v1.json`. Interpret the run in this order.

1. Stop as invalid if provenance, coverage, exact-environment, or rank-96
   full-rank replay fails.
2. Treat the globally trace-balanced exact basis as primary and the raw equal
   exact basis as an ablation. Call a strong go only if the primary basis at
   rank 64 beats local radial SVD, the prior first-branch exact basis, and the
   prior total-energy-gradient basis under the paired rule, while reaching
   force RMSE at most `0.001 eV/A` and energy error at most `0.0001 eV/atom`.
3. Call a weak go if the paired rank-64 baseline comparisons pass but the
   absolute thresholds do not.
4. If only rank 80 reaches the relative and absolute criteria, including its
   paired intervals and Holm-adjusted sign-flip tests, continue as a
   direction-selection result and explicitly report that the representation is
   not compact.
5. Trigger no-go for this objective and measure if the primary exact method
   fails to beat both local radial SVD and the total-energy-gradient method at
   ranks 64 and 80. The raw equal ablation cannot rescue that primary no-go.

Reverse-spectrum and five canonical-random methods are sanity controls. They do
not change the preregistered go or no-go decisions.

## Caveats

The exact environment is exact only for its frozen discrete radial and angular
measure. Held-out fidelity remains empirical because the same projector is
inserted simultaneously at every applicable node. Passing does not establish a
faster evaluator, storage compression, cross-checkpoint recurrence, or a
physical mechanism.
