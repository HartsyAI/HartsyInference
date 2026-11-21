# Vulkan dense F32 attention: scalar route against the query-tiled kernel

When this was taken `numerics.vkTiledAttention` was `off` by default, and its description said it stayed off "until
the regression A/B for the default flip is recorded". This is that A/B on one card. The default has since changed to
`auto`; [the last section](#the-default-flip) records that change and the reruns behind it. In the tables above it,
"default" means the knob `off`, and "tiled" means the kernel at every head dim, which is now the value `all`.

## What was compared

Dense F32 attention through `IBackend.ScaledDotProductAttention` on Vulkan, head-major operands already on the device:

- **Default**: the knob off. The call takes `flash.scalar` (row A.2) up to head dim 128 and `naive` (A.3) above it.
- **Tiled**: the knob on. The call takes `tiled.f32` (A.2t), the query-tiled F32 kernel.

Same call, same device buffers, one backend, the two arms alternated run by run. Harness:
`tests/HartsyInference.LLM.Tests/Parity/VulkanTiledAttentionAbTests.cs` (category `GpuBenchmark`, which the CPU
lane and the GPU lane both leave out). It fails if the tiled arm did not take `tiled.f32`, so a device that cannot
run the kernel cannot report a comparison of the default row with itself.

- Card: RTX 3060 12 GB, shared with a desktop session; each run started only with the card under 10% utilisation.
- Each figure is the median of 7 calls after 2 warm-ups. The test was run three times; the ranges below are over those
  three medians.
- Each time is the wall time of the call and a `Sync`, so it includes allocating the output buffer on the device and
  the round trip, for both arms. That is small beside the gaps at 256 tokens and above and a visible part of the
  figures under 2 ms.
- This is descriptive. It is not the frozen protocol of `protocol.json` (no paired sessions, no quiet-window evidence).

Expectation written in the test before the first run: tiled at least 3 times faster at 1,024 by 1,024 tokens, possibly
slower on the smallest shapes, both outputs within 2e-5 of the host reference.

## Result

8 heads unless stated. Times in milliseconds.

| Call | Default row | Default | Tiled | Default over tiled |
| --- | --- | --- | --- | --- |
| Self, 64 tokens, D 64 | `flash.scalar` | 0.29 to 0.34 | 0.11 to 0.17 | 2.0 to 2.6 |
| Self, 256 tokens, D 128 | `flash.scalar` | 5.4 to 5.7 | 0.44 to 0.50 | 10.8 to 12.8 |
| Self, 1,024 tokens, D 64 | `flash.scalar` | 43.6 to 46.8 | 1.42 to 1.50 | 30.1 to 31.2 |
| Self, 1,024 tokens, D 128 | `flash.scalar` | 78.5 to 81.8 | 4.14 to 4.17 | 18.8 to 19.8 |
| Self, 1,024 tokens, D 128, 2 key/value heads | `flash.scalar` | 79.0 to 81.2 | 4.03 to 4.08 | 19.6 to 19.9 |
| Self, 1,024 tokens, D 128, additive bias | `flash.scalar` | 82.4 to 91.4 | 4.14 to 6.07 | 15.1 to 19.9 |
| Self, 512 tokens, D 128, batch 2 | `flash.scalar` | 42.3 to 44.4 | 2.26 to 3.83 | 11.6 to 19.1 |
| Self, 4,096 tokens, D 64, 16 heads | `flash.scalar` | 1,407 to 1,422 | 38.1 to 38.8 | 36.7 to 37.0 |
| Cross, 4,096 by 77, D 128 | `flash.scalar` | 27.8 to 28.4 | 1.38 to 1.46 | 19.3 to 20.3 |
| Cross, 4,096 by 512, D 128 | `flash.scalar` | 158.3 to 160.7 | 7.42 to 7.92 | 20.1 to 21.6 |
| One row over 4,096 keys, D 128 | `flash.scalar` | 2.48 to 2.50 | 2.04 to 2.07 | 1.2 |
| **Self, 1,024 tokens, D 256** | **`naive`** | **6.7 to 6.9** | **14.3 to 14.7** | **0.45 to 0.48** |

Numerics, largest absolute difference over the whole output:

- From the host reference (taken for every call but the 4,096-token self-attention, which is too large for it here):
  the default row is at most 1.2e-7 away, the tiled kernel at most 1.6e-7. Both are far inside 2e-5.
- Between the two arms: at most 1.7e-7 on every call, including the one without a host reference.

## Reading

- **Up to head dim 128 the tiled kernel is faster on every shape measured**, by 11 to 37 times at 256 tokens and
  above, by 2 times at 64 tokens, and by 1.2 times for a single query row. The expectation of at least 3 times at
  1,024 tokens held (19 and 30 times); the guess that it might lose on the smallest shapes did not (it wins there too).
- **At head dim 256 the tiled kernel is about twice as slow as the default.** The default there is not `flash.scalar`,
  which stops at head dim 128, but `naive`, which materializes the scores. With the knob on, `tiled.f32` is listed
  ahead of `naive` and takes the call. Turning the knob on as it is coded today would therefore be a regression at
  head dims 129 to 256.
- `naive` at head dim 256 (6.8 ms) is itself faster than `flash.scalar` at head dim 128 (80 ms) for the same token
  counts. I did not measure `naive` at head dim 128 or 64, because no knob selects it there; it is possible that it
  also beats `flash.scalar` at those sizes, at the cost of materializing the scores.
- The numbers do not separate the kernels numerically: both are exact F32 and agree to the last few bits.

## What this supports, and what it does not

It supports turning the tiled kernel on by default **for head dims up to 128**, with the head-dim-256 bucket left
behind `naive` unless a call's scores would not fit in memory (the case the tiled kernel was first written for). That is
a change to which row a call takes, so it needs:

- the same A/B on the RTX 4090 and, if one can be had, on a non-NVIDIA device, since the tiled kernel's tile sizes were
  chosen on NVIDIA cards;
- the route matrix and the real-generation checks the plan requires for a planner or admission change
  (`proposed-queue.md`, evidence cadence), on every Vulkan workload whose dense F32 attention would move;
- a decision on the head-dim-256 bucket.

It does not show any whole-model speedup. A model spends only part of its time in dense F32 attention on Vulkan, and
calls that permit F16 already take the cooperative-matrix row (0.6 ms for the 1,024-token D 128 call on this card),
which is faster than either arm here.

Not measured: peak memory of each arm, masks other than one additive bias, token-major calls, F16 or BF16 operands
(the tiled row is F32 only), and any device but this one.

## The default flip

The default of `numerics.vkTiledAttention` is now `auto`, and the knob has three values:

- `auto`: a dense call whose operands and output are all F32 takes `tiled.f32` up to head dim 128. At head dims 129
  to 256 it stays on `naive`, unless it has a bias and its score matrix is past 1 GiB.
- `all`: the tiled kernel at every head dim up to 256, which is what `auto` meant when the table above was taken.
- `off`: `flash.scalar` up to head dim 128 and `naive` above it, as before.

Why that rule at head dims 129 to 256: `naive` is about twice as fast there, and it tiles the query axis itself once an
unmasked call's scores pass 1 GiB, so it never has to hold more. With a bias it cannot tile, and holds two buffers of
the whole score matrix. That is the one case the default gives to the tiled kernel, at the same 1 GiB (one constant in
the backend, read by both). How a large unmasked call on self-tiled `naive` compares with the tiled kernel was not
measured, and neither was any call past 1 GiB of scores.

### Reruns on the branch

`VulkanTiledAttentionAbTests` again, the arms now `off` and `all`, same card. Tolerance set before the reruns (in the
session, not in the repo): every figure of 1 ms or more within 25% of the range above, the tiled kernel at least 10
times faster at 256 tokens and above for head dims up to 128, and about half the speed of `naive` at head dim 256.

**None of the reruns had the card to itself.** It is shared with three other agent sessions. The first three runs
overlapped another session's GPU test host for their whole length; two of them are scattered far from the table above
(the tiled kernel at 1,024 tokens, D 128, read 9.8 and 23.1 ms against 4.2) and are not used. The next three started
only when no other test host of this repository was on the card, and sampled for one every 3 seconds: another test
host was present in 25 of 60, 43 of 66 and 47 of 64 samples. Those three are the table below. Read it as "the figures
above still hold where the card was quiet enough", not as a clean replication.

| Call | Row with `off` | `off` | `all` (tiled) | `off` over tiled | Merged range held? |
| --- | --- | --- | --- | --- | --- |
| Self, 64 tokens, D 64 | `flash.scalar` | 0.35 to 0.52 | 0.18 to 0.25 | 1.7 to 2.9 | under 1 ms, not judged |
| Self, 256 tokens, D 128 | `flash.scalar` | 5.7 to 7.7 | 0.57 to 1.70 | 4.5 to 10.6 | **no**: under 10 times in two runs |
| Self, 1,024 tokens, D 64 | `flash.scalar` | 43.6 to 53.3 | 1.44 to 1.86 | 25.7 to 30.4 | yes |
| Self, 1,024 tokens, D 128 | `flash.scalar` | 79.7 to 82.2 | 4.19 to 4.38 | 18.7 to 19.0 | yes |
| Self, 1,024 tokens, D 128, 2 key/value heads | `flash.scalar` | 78.8 to 89.6 | 4.03 to 4.51 | 19.5 to 19.9 | yes |
| Self, 1,024 tokens, D 128, additive bias | `flash.scalar` | 85.8 to 110.5 | 4.18 to 8.85 | 9.7 to 23.1 | **no**: 9.7 times in one run |
| Self, 512 tokens, D 128, batch 2 | `flash.scalar` | 39.7 to 48.6 | 2.15 to 3.19 | 14.4 to 18.5 | yes |
| Self, 4,096 tokens, D 64, 16 heads | `flash.scalar` | 1,398 to 1,652 | 37.6 to 44.2 | 37.1 to 37.4 | yes |
| Cross, 4,096 by 77, D 128 | `flash.scalar` | 26.0 to 34.2 | 1.41 to 2.13 | 16.1 to 18.5 | **no**: tiled 2.13 ms in one run |
| Cross, 4,096 by 512, D 128 | `flash.scalar` | 158.3 to 198.7 | 9.29 to 12.04 | 16.5 to 19.4 | **no**: tiled 9.3 to 12.0 ms against 7.4 to 7.9 |
| One row over 4,096 keys, D 128 | `flash.scalar` | 2.69 to 4.16 | 2.35 to 4.86 | 0.86 to 1.19 | **no**: tiled slower in one run |
| Self, 1,024 tokens, D 256 | `naive` | 7.1 to 9.8 | 15.1 to 18.3 | 0.46 to 0.54 | **no**: `naive` 9.8 ms in one run; the ratio held |

The misses are all in the direction of a disturbed card (slower, never faster), and the least disturbed of the three
runs is inside the tolerance on every row but one: self-attention at 256 tokens, where the tiled arm read 1.27 ms
against 0.44 to 0.50. The single query row is the one call where the tiled kernel's lead (1.2 times) is smaller than
the noise on this card: it read 0.86 in one of these runs and 0.66 in one of the unused ones.

Numerics were the same in every run: the two arms differ by at most 1.7e-7, the `off` row is at most 1.2e-7 from the
host reference and the tiled kernel at most 1.6e-7.

### Head dim 256 under the new default

A case added for the flip (`HeadDim256UnderTheDefaultStaysOnTheMaterializedRow`): self-attention, 8 heads, 1,024
tokens, D 256, the knob at `off` and at `auto`, alternated. Expectation written in the test before the first run: both
take `naive` (A.3), identical outputs, medians within 20% of each other.

| Run | `off` | `auto` | Row, both arms | Outputs |
| --- | --- | --- | --- | --- |
| 1 | 6.73 | 6.79 | `naive` | identical |
| 2 | 9.03 | 7.67 | `naive` | identical |
| 3 | 7.05 | 7.03 | `naive` | identical |

So the regression the first A/B found at head dim 256 (14.5 ms against 6.8) does not come with the new default.

### Routes and parity

- `VulkanTiledAttentionRouteTests` plans 448 dense head-major calls (head dims 32, 64, 96, 128, 192, 256, 320; F32, F16,
  BF16 and F32-over-F16-keys operands; with and without the F16 permission, a bias, grouped heads, batch 2) under each
  knob value against rows written out in the test. On this card 48 move under the default, every one an all-F32 call
  leaving `flash.scalar` at head dim 128 or below. The cooperative-matrix row keeps F32 calls that carry the F16
  permission at head dims 64 and 128, so on a device without that kernel those move too.
- `VulkanTiledAttentionParityTests` runs the tiled row against the host reference in the GPU lane, which nothing did
  before (this A/B is a benchmark, outside both lanes): 15 head-major cases across the three buckets with token counts
  off the tile heights, grouped heads, batch 2, per-query, per-head and key-only biases and a fully excluded row, at
  most 1.8e-7 relative L2 against the contract's 2e-5. A token-major call, F32 or BF16, reaches the same row through
  the permute.

### Still not shown

No RTX 4090 run and no non-NVIDIA device: the tile sizes were chosen on NVIDIA cards and every figure here is from one
RTX 3060. No real generation before and after, so no whole-model time or output comparison. No peak memory. No quiet
card for the reruns.
