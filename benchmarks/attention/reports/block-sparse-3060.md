# Block-sparse attention on an RTX 3060 (2026-11-27)

CUDA `IBackend.BlockSparseAttention` beside `ScaledDotProductAttention` on the same device-resident operands, at token grids of
LongCat-Video's 720p refinement, head dim 128, sparsity 0.9375 and 4×4×4 blocks.

- Commit `c665f9ca8`, RTX 3060 12 GB, driver 610.88, Windows 11, .NET 10 Release, default engine settings (TF32 allowed for
  dense F32, cuDNN on). No other test or inference process on the card; its processor was busy with other work.
- One untimed call, then the median of three synchronized calls. Output in the operands' dtype.
- `dotnet test tests/HartsyInference.Core.Tests -c Release --filter FullyQualifiedName~BlockSparseAttentionSpeedTests`

| Token grid (T×H×W) | Tokens | Heads | Key blocks read | Operands | Block-sparse | Dense | Dense / block-sparse |
|---|---|---|---|---|---|---|---|
| 12×44×80 | 42,240 | 32 | 41 of 660 | F32 | 878 ms | 9,280 ms | 10.6 |
| 12×44×80 | 42,240 | 32 | 41 of 660 | BF16 | 834 ms | 8,373 ms | 10.0 |
| 48×44×80 | 168,960 | 8 | 165 of 2640 | F32 | 3,478 ms | 31,828 ms | 9.2 |
| 48×44×80 | 168,960 | 8 | 165 of 2640 | BF16 | 3,492 ms | 34,143 ms | 9.8 |

The 48-frame grid runs 8 heads rather than the model's 32: four operands of 32 heads do not fit in 12 GB. Which dense kernel
row served each call was not recorded.

The block-sparse kernel computes in F32 whatever it reads. BF16 operands save memory; their times differ from F32 by 5
percent or less. A query block reads 1/16 of the keys and the time falls 9 to 11 times; why not 16 was not measured.
Its workspace beside the operands is the block means and the selection, 34 MiB at the 48-frame, 8-head size (computed from the
shapes).

## CUDA tile layout

The attention stage alone, F32, 12×44×80 with 8 heads and 41 blocks read, synthetic selection, median of three. Each row is one
build of the kernel with other tile constants.

| Query rows × key rows a block | Threads sharing 4 query rows | Time |
|---|---|---|
| 64 × 64 | 16 | 282 ms |
| 64 × 64 | 32 (adopted for head dim 32 and up) | 207 ms |
| 64 × 16 | 16 | 219 ms |
| 64 × 32 | 32 | 257 ms |
| 32 × 32 | 16 | 433 ms |

Groups of 2 query rows instead of 4 gave 288 ms (64 × 64) and 320 ms (64 × 32). Head dim 16 keeps 16 threads a group, which is
all its channels split into.

## Vulkan

`IBackend.BlockSparseAttention` on Vulkan, same card, driver, options and head dim, output in the operands' dtype.

- Commit `39a037a11`, Windows 11, .NET 10 Release. One untimed call, then the median of three synchronized calls.
- One case per run, smallest first:
  `dotnet test tests/HartsyInference.Core.Tests -c Release --filter "FullyQualifiedName~BlockSparseAttentionSpeedTests.VulkanBlockSparse&DisplayName~heads: 32"`

| Token grid (T×H×W) | Tokens | Heads | Key blocks read | F32 | BF16 |
|---|---|---|---|---|---|
| 4×44×80 | 14,080 | 1 | 13 of 220 | 20.7 ms | 16.7 ms |
| 4×44×80 | 14,080 | 4 | 13 of 220 | 45.0 ms | 43.4 ms |
| 12×44×80 | 42,240 | 8 | 41 of 660 | 795 ms | 784 ms |
| 12×44×80 | 42,240 | 32 | 41 of 660 | 3,182 ms | 3,179 ms |

At 12×44×80 with 32 heads Vulkan takes 3.6 times CUDA's time in the table above (CUDA measured at `c665f9ca8`); why was not
measured. The attention pass goes in 14 submissions at 8 heads and 55 at 32 (computed from the submission budget), and the
driver logged no event during these runs. The outputs were not compared with the reference. The 48-frame grid was not timed.

Dense attention is not timed on Vulkan. On this card it lost the device (`ErrorDeviceLost`) at 12×44×80 with 8 heads on F32
operands and at 4×44×80 with 1 head on BF16 operands; the cause is not established.
