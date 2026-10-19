using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>3-axis rotary position embedding for ERNIE-Image (Baidu, Apache-2.0). Splits the head dimension across 3 axes <c>(text_pos, y, x)</c> with default sizes <c>(32, 48, 48)</c> and theta=256. The freqs tensor uses **non-interleaved Megatron-style "rotate_half"**: per-axis angles are duplicated to form <c>[θ0,θ0,θ1,θ1,...]</c>, and rotation pairs token features at indices <c>(i, i + halfDim)</c> rather than the Flux interleaved <c>(2k, 2k+1)</c> layout. Cannot reuse <see cref="FluxRope"/> or <see cref="ZImageRope"/>.
///
/// Image tokens at grid <c>(row, col)</c> in batch <c>b</c> get position <c>(text_lens[b], row, col)</c> — image RoPE positions are offset by the per-batch text length, which is non-standard. Text token at index <c>t</c> gets position <c>(t, 0, 0)</c>. The diffusers reference (<c>transformer_ernie_image.py</c>) concatenates as <c>[image_ids; text_ids]</c> — image tokens come FIRST in the sequence — so this class produces position IDs and a precomputed freqs tensor in that same order.</summary>
public sealed unsafe class ErnieImageRope
{
    private readonly int[] _axesDim;
    private readonly int _headDim;
    private readonly RopeFrequencyPlan _plan;

    /// <summary>Creates a 3-axis ERNIE-Image RoPE.</summary>
    /// <param name="axesDim">Per-axis dimensions <c>(text_pos, y, x)</c>. Must sum to head_dim. Default <c>[32, 48, 48]</c> (sums to 128).</param>
    /// <param name="theta">RoPE base frequency. Default 256 — much smaller than Flux's 10000.</param>
    public ErnieImageRope(int[]? axesDim = null, float theta = 256.0f)
    {
        _axesDim = axesDim ?? [32, 48, 48];
        if (_axesDim.Length != 3)
            throw new ArgumentException($"ErnieImageRope expects 3 axes, got {_axesDim.Length}.", nameof(axesDim));
        for (int i = 0; i < _axesDim.Length; i++)
        {
            if (_axesDim[i] % 2 != 0)
                throw new ArgumentException($"Axis {i} dim must be even (got {_axesDim[i]}).", nameof(axesDim));
        }
        // The reference widens the float base to double for every frequency, so the plan's base is that widened value.
        _plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = _axesDim, Theta = theta });
        _headDim = _plan.HeadDim;
    }

    /// <summary>Total head dimension (sum of axes, 128 by default).</summary>
    public int HeadDim => _headDim;

    /// <summary>Number of axes (always 3 for ERNIE-Image).</summary>
    public int NumAxes => _axesDim.Length;

    /// <summary>Builds the <c>[B, totalSeq, 3]</c> position-ID tensor for the concatenated <c>[image, text]</c> sequence. Image tokens come first (matches <c>transformer_ernie_image.py:365</c>'s <c>cat([img_sbh, text_sbh])</c>). Image token <c>(row, col)</c> in batch <c>b</c> gets <c>(text_lens[b], row, col)</c>; text token at index <c>t</c> gets <c>(t, 0, 0)</c>.</summary>
    /// <param name="batch">Batch size.</param>
    /// <param name="textLens">Per-batch real (non-padded) text token counts. Length must equal <paramref name="batch"/>.</param>
    /// <param name="gridH">Image grid height (latent_h / patch_size).</param>
    /// <param name="gridW">Image grid width.</param>
    /// <param name="textMax">Padded text sequence length (<c>Tmax</c>). Padded slots also get monotonically increasing positions because the diffusers reference uses <c>arange(Tmax)</c> rather than only the real prefix.</param>
    public static Tensor BuildPositionIds(int batch, int[] textLens, int gridH, int gridW, int textMax)
    {
        if (textLens.Length != batch)
            throw new ArgumentException($"textLens length {textLens.Length} != batch {batch}.", nameof(textLens));

        int nImg = gridH * gridW;
        int totalSeq = nImg + textMax;

        TensorShape shape = new TensorShape(batch, totalSeq, 3);
        Tensor posIds = new Tensor(shape, DType.F32);
        float* p = (float*)posIds.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            float textLenF = textLens[b];

            // [0 .. nImg) — image tokens at (text_lens[b], row, col)
            for (int row = 0; row < gridH; row++)
            {
                for (int col = 0; col < gridW; col++)
                {
                    int idx = row * gridW + col;
                    long off = ((long)b * totalSeq + idx) * 3;
                    p[off + 0] = textLenF;
                    p[off + 1] = row;
                    p[off + 2] = col;
                }
            }

            // [nImg .. nImg+textMax) — text tokens at (t, 0, 0). Diffusers uses arange(Tmax), so even pad slots get monotonic frames.
            for (int t = 0; t < textMax; t++)
            {
                long off = ((long)b * totalSeq + nImg + t) * 3;
                p[off + 0] = t;
                p[off + 1] = 0f;
                p[off + 2] = 0f;
            }
        }

        return posIds;
    }

    /// <summary>Computes the cos/sin freqs tensor used by <see cref="ApplyRotaryEmbGpu"/>. Output shape is <c>[B, totalSeq, head_dim]</c> packed as <c>[cos_0..cos_{D-1}, sin_0..sin_{D-1}]</c> per token along the last dim — but stored as two separate halves of length <c>head_dim</c>: a cos block then a sin block, total length <c>2 * head_dim</c>. We pack both halves into one tensor so the GPU upload path is a single buffer; <see cref="ApplyRotaryEmbGpu"/> indexes them as <c>cos = freqs[..., :head_dim]</c> and <c>sin = freqs[..., head_dim:]</c>.</summary>
    public Tensor BuildFreqs(Tensor posIds)
    {
        int batch = (int)posIds.Shape[0];
        int totalSeq = (int)posIds.Shape[1];
        int numAxes = (int)posIds.Shape[2];
        if (numAxes != _axesDim.Length)
            throw new ArgumentException($"posIds last dim {numAxes} != configured axes {_axesDim.Length}.", nameof(posIds));

        // Per-axis angles concatenate into headDim/2 pairs (torch.cat([rope(ids[...,a], axes_dim[a], theta) ...], -1)), and
        // the stack-and-reshape duplicates each pair into [θ0, θ0, θ1, θ1, ...] — the interleaved-pair row layout.
        int tokens = checked(batch * totalSeq);
        int rowLength = checked(tokens * _headDim);
        // Batched generation turns the transformer's freqs cache off and rebuilds every forward, so the rows are
        // unmanaged scratch.
        nuint bytes = (nuint)Math.Max(rowLength, 1) * sizeof(float);
        float* cosScratch = (float*)NativeMemory.AlignedAlloc(bytes, 64);
        float* sinScratch = null;
        try
        {
            sinScratch = (float*)NativeMemory.AlignedAlloc(bytes, 64);
            Span<float> cosRows = new(cosScratch, rowLength);
            Span<float> sinRows = new(sinScratch, rowLength);
            MultiAxisRopeTables.BuildRows(_plan, new ReadOnlySpan<float>(posIds.DataPointer, checked(tokens * numAxes)), tokens,
                RopeAngleOrder.MultiplyByInverse, RopeTableLayout.InterleavedPair, 0, RopeTablePadding.Identity, cosRows, sinRows);

            // Packed per token as [cos(headDim) | sin(headDim)].
            Tensor freqs = new Tensor(new TensorShape(batch, totalSeq, 2 * _headDim), DType.F32);
            float* outPtr = (float*)freqs.DataPointer;
            for (int t = 0; t < tokens; t++)
            {
                long outBase = (long)t * 2 * _headDim;
                cosRows.Slice(t * _headDim, _headDim).CopyTo(new Span<float>(outPtr + outBase, _headDim));
                sinRows.Slice(t * _headDim, _headDim).CopyTo(new Span<float>(outPtr + outBase + _headDim, _headDim));
            }
            return freqs;
        }
        finally
        {
            NativeMemory.AlignedFree(cosScratch);
            NativeMemory.AlignedFree(sinScratch);
        }
    }

    /// <summary>Applies the non-interleaved Megatron-style rotation to Q/K on-device: slices the packed
    /// <c>[B, S, 2·headDim]</c> freqs (from <see cref="BuildFreqs"/>) into <c>cos = freqs[..., :headDim]</c> /
    /// <c>sin = freqs[..., headDim:]</c> and applies <c>x·cos + rotate_half(x)·sin</c> with
    /// <c>rotate_half(x)=cat(-x[half:], x[:half])</c> via <see cref="IBackend.ApplyRope"/>. Keeps Q/K
    /// device-resident (no per-block D2H/H2D). q/k are <c>[B, S, numHeads, headDim]</c> (pre-permute,
    /// matching <c>transformer_ernie_image.py</c>'s <c>unflatten(-1, (heads, -1))</c> layout), modified in place.</summary>
    public void ApplyRotaryEmbGpu(IBackend backend, Tensor q, Tensor k, Tensor freqs)
    {
        int batch = (int)q.Shape[0];
        int seqLen = (int)q.Shape[1];
        TensorShape csShape = new TensorShape(batch, seqLen, _headDim);
        Tensor cos = new Tensor(csShape, DType.F32);
        Tensor sin = new Tensor(csShape, DType.F32);
        backend.SliceLastDim(cos, freqs, 0);          // freqs[..., 0 : headDim]
        backend.SliceLastDim(sin, freqs, _headDim);   // freqs[..., headDim : 2·headDim]
        backend.ApplyRope(q, k, cos, sin);
        cos.Dispose();
        sin.Dispose();
    }
}
