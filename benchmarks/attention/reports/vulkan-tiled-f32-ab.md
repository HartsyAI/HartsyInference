# Vulkan dense F32 attention: scalar route against the query-tiled kernel

`numerics.vkTiledAttention` is `off` by default, and its description says it stays off "until the regression A/B for the
default flip is recorded". This is that A/B on one card. It does not change the default.

## What was compared

Dense F32 attention through `IBackend.ScaledDotProductAttention` on Vulkan, head-major operands already on the device:

- **Default**: the knob off. The call takes `flash.scalar` (row A.2) up to head dim 128 and `naive` (A.3) above it.
- **Tiled**: the knob on. The call takes `tiled.f32` (A.2t), the query-tiled F32 kernel.

Same call, same device buffers, one backend, the two arms alternated run by run. Harness:
`tests/HartsyInference.LLM.Tests/Parity/VulkanTiledAttentionAbTests.cs` (category `Timing`, outside the GPU lane).

- Card: RTX 3060 12 GB, shared with a desktop session; each run started only with the card under 10% utilisation.
- Each figure is the median of 7 calls after 2 warm-ups. The test was run three times; the ranges below are over those
  three medians.
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
