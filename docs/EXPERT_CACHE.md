# Expert residency policy

`IAdaptiveResidencyPolicy` (`src/HartsyInference.Core/Moe/Residency/`) chooses which resident expert to evict. It is
standalone: it never reads or writes the cache. `ExpertCacheBase` reports every routed access, every insert after it
lands and every eviction to an attached policy, and asks it for a victim when a slot is needed; with no policy attached
the cache falls back to its own segmented LRU. `DecayedLfuHysteresisPolicy` is the one implementation. `MoeExpertOffload`
attaches it to the cache and also reads its scores to decide whether a miss is worth an upload; the CUDA offload path
constructs it with the decay and hysteresis constants in `CudaMoeOffload`.

- The clock advances by one per routed access. Every score decays by `decayPerStep` per step, kept lazily as
  (value, epoch) so no pass over all experts is needed. An access adds 1 to the expert's decayed score.
- Hysteresis works through an incumbent, the victim chosen last. The incumbent stays the victim unless another candidate
  scores below it by more than `hysteresisMargin`, as a fraction of the incumbent's score. An access to the incumbent, or
  its eviction, clears it, so near-equal experts do not flip on every miss.
- The policy is deterministic: the same call sequence gives the same victims, and ties go to the earlier candidate. After
  warm-up the hot methods allocate nothing.
