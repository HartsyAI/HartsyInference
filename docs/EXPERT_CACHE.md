# Expert residency: telemetry, policies, and trace replay

Status: standalone components. **The residency policies below are not yet wired into `ExpertCacheBase`.** The expert cache
still uses its own eviction logic, and nothing in this change alters it. Integration is a later change. The components here
exist so policies can be compared on recorded or synthetic routing traces before any cache behavior depends on them.

Code lives in `src/HartsyInference.Core/Moe/Telemetry/` and `src/HartsyInference.Core/Moe/Residency/`. Tests live in
`tests/HartsyInference.Core.Tests/Moe/Telemetry/` and `tests/HartsyInference.Core.Tests/Moe/Residency/`.

## Routing telemetry

`RoutingTelemetry` is a fixed-capacity recorder. Every array is allocated in the constructor, and `Record` and `BeginStep`
do not allocate.

- Per-layer counters (`LayerRoutingCounters`): routed, hits, misses, bytes moved, CPU runs, GPU runs, and a hit rate.
- Per-expert frequency within a bank, indexed by layer and expert.
- A ring of `RoutingStepRecord` values, one per routed access, holding the step, the `ExpertKey` (bank included), bytes
  moved, and outcome flags (`RoutingEventFlags`: hit, GPU run). Old records are overwritten once the ring is full.

Bytes moved are what the caller reports for the access, normally zero on a hit. The telemetry is single-threaded.

## Residency policies

Policies implement `IAdaptiveResidencyPolicy`, which is independent of `ExpertCacheBase`. The caller reports:

- `NoteAccess(key)` for every routed access, hit or miss, before any eviction that access causes.
- `NoteInsert(key)` after an expert lands in a slot.
- `NoteEvict(key)` after an expert leaves a slot.
- `ChooseVictim(candidates)` when a slot is needed, with every candidate currently resident.

Each policy is deterministic: the same call sequence gives the same victims. Ties go to the earlier candidate or to the least
recently used entry, so no dictionary iteration order affects a decision. After warm-up, the four hot methods allocate nothing.

- **`SegmentedLruPolicy(protectedCapacity)`**: residents start on probation. A hit on a probationary resident promotes it to
  the protected segment. When protected exceeds its capacity, its least recently used member is demoted. Victims come from
  probation first (least recently used), then from protected.
- **`LfuPolicy`**: perfect least-frequently-used. Counts persist after eviction and never age. The victim is the lowest count,
  with ties going to the least recently used.
- **`DecayedLfuHysteresisPolicy(decayPerStep, hysteresisMargin)`**: the clock advances by one per access (one step). Each
  score is multiplied by `decayPerStep` per step, kept lazily as (value, epoch). Hysteresis works through an incumbent, the
  victim chosen last. The incumbent stays the victim unless some other candidate scores below it by more than the margin, as a
  fraction of the incumbent's score. An access to the incumbent, or its eviction, clears it.

`SlotCacheSimulator` drives a policy over a fixed slot budget. Misses fill free slots first. When the cache is full, the
policy picks the victim. The simulator keeps an FNV-1a digest of every victim, so two replays can be compared exactly.

## Trace format

Traces are recorded and replayed in a versioned binary format, little-endian (`ExpertTraceCodec`).

- Header, 16 bytes: magic `HMTR`, version `u16` (currently 1), record size `u16` (8), record count `u64`, reserved `u32`
  (must be zero).
- Body: one 8-byte record per access, `ExpertTraceRecord(layer u16, expert u16, bytes u32)`.

Readers reject a bad magic, an unsupported version, a wrong record size, non-zero reserved bytes, a body shorter than the
count says, and trailing bytes. Each rejection throws `ExpertTraceFormatException`.

## Harness and synthetic traces

- `SyntheticTraceGenerator.Zipf(seed, layers, expertsPerLayer, steps, topK, exponent, bytesPerExpert)` builds a trace with
  `SplitMix64`. Each layer has its own seeded ranking of experts, and each step draws `topK` distinct experts per layer from a
  Zipf distribution over that ranking. The same seed always gives the same trace.
- `TraceReplayHarness.Replay(trace, policy, slotBudget)` replays a trace through a fresh policy and returns a `ReplayResult`:
  accesses, hits, misses, evictions, bytes moved, the decision digest, and the hit rate. Use a fresh or reset policy for
  each comparison.

Hit rates on the seeded traces (4 layers, 64 experts per layer, 2000 steps, top-4, 1 MiB per expert, seed 42):

| Zipf exponent | Slots | SegmentedLru | Lfu | DecayedLfuHysteresis (0.995, 0.1) |
|---|---|---|---|---|
| 1.0 | 32 | 0.3809 | 0.5111 | 0.4688 |
| 1.2 | 32 | 0.4725 | 0.6049 | 0.5698 |
| 1.0 | 96 | 0.6716 | 0.7629 | 0.6975 |

These traces are stationary, so perfect LFU leads. Decayed LFU gives up some of that history to adapt to drift, which a
stationary trace does not reward. These numbers show behavior on synthetic input only. They say nothing yet about real
routing or about the cache.

## Tests

- Determinism: replaying the same trace through a fresh policy gives identical results and digests.
- Allocation: `Record`, `NoteAccess`, `ChooseVictim`, `NoteEvict`, and `NoteInsert` allocate zero bytes after warm-up,
  measured with `GC.GetAllocatedBytesForCurrentThread`.
- Hot experts: an expert routed every other access, against a cold scan, stays resident under all three policies.
- Round trip and corruption: traces round-trip exactly, and corrupt, truncated, or extended input is rejected.
