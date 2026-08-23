using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>3-axis rotary position embedding for ERNIE-Image (Baidu, Apache-2.0). Splits the head dimension across 3 axes <c>(text_pos, y, x)</c> with default sizes <c>(32, 48, 48)</c> and theta=256. The freqs tensor uses **non-interleaved Megatron-style "rotate_half"**: per-axis angles are duplicated to form <c>[θ0,θ0,θ1,θ1,...]</c>, and rotation pairs token features at indices <c>(i, i + halfDim)</c> rather than the Flux interleaved <c>(2k, 2k+1)</c> layout. Cannot reuse <see cref="FluxRope"/> or <see cref="ZImageRope"/>.
///
/// Image tokens at grid <c>(row, col)</c> in batch <c>b</c> get position <c>(text_lens[b], row, col)</c> — image RoPE positions are offset by the per-batch text length, which is non-standard. Text token at index <c>t</c> gets position <c>(t, 0, 0)</c>. The diffusers reference (<c>transformer_ernie_image.py</c>) concatenates as <c>[image_ids; text_ids]</c> — image tokens come FIRST in the sequence — so this class produces position IDs and a precomputed freqs tensor in that same order.</summary>
public sealed unsafe class ErnieImageRope
{
    private readonly int[] _axesDim;
    private readonly float _theta;
    private readonly int _headDim;

    /// <summary>Creates a 3-axis ERNIE-Image RoPE.</summary>
    /// <param name="axesDim">Per-axis dimensions <c>(text_pos, y, x)</c>. Must sum to head_dim. Default <c>[32, 48, 48]</c> (sums to 128).</param>
    /// <param name="theta">RoPE base frequency. Default 256 — much smaller than Flux's 10000.</param>
    public ErnieImageRope(int[]? axesDim = null, float theta = 256.0f)
    {
        _axesDim = axesDim ?? [32, 48, 48];
        if (_axesDim.Length != 3)
            throw new ArgumentException($"ErnieImageRope expects 3 axes, got {_axesDim.Length}.", nameof(axesDim));
        _theta = theta;
        _headDim = 0;
        for (int i = 0; i < _axesDim.Length; i++)
        {
            if (_axesDim[i] % 2 != 0)
                throw new ArgumentException($"Axis {i} dim must be even (got {_axesDim[i]}).", nameof(axesDim));
            _headDim += _axesDim[i];
        }
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

        int halfDim = _headDim / 2;
        TensorShape outShape = new TensorShape(batch, totalSeq, 2 * _headDim);
        Tensor freqs = new Tensor(outShape, DType.F32);
        float* posPtr = (float*)posIds.DataPointer;
        float* outPtr = (float*)freqs.DataPointer;

        // Per-axis omega tables: freq_k = 1 / theta^(2k / axis_dim), k in [0, axis_dim/2).
        int maxPairs = 0;
        for (int a = 0; a < numAxes; a++)
            maxPairs = Math.Max(maxPairs, _axesDim[a] / 2);
        Span<double> omega = stackalloc double[maxPairs];
        // Allocate once outside the per-token loop to avoid CA2014 stackalloc-in-loop and the
        // associated stack-overflow risk for large halfDim or large totalSeq.
        Span<double> halfAngles = stackalloc double[halfDim];

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < totalSeq; s++)
            {
                long posOff = ((long)b * totalSeq + s) * numAxes;
                long outBase = ((long)b * totalSeq + s) * (2L * _headDim);

                // ── 1. Build the halfDim-long angle vector by concatenating per-axis angles ──
                // (Equivalent to Python's `torch.cat([rope(ids[...,a], axes_dim[a], theta) for a in range(3)], -1)`.)
                int writeOffset = 0;
                for (int a = 0; a < numAxes; a++)
                {
                    int axisDim = _axesDim[a];
                    int axisPairs = axisDim / 2;
                    double pos = posPtr[posOff + a];
                    for (int k = 0; k < axisPairs; k++)
                    {
                        double scale = (double)(2 * k) / axisDim;
                        omega[k] = 1.0 / Math.Pow(_theta, scale);
                        halfAngles[writeOffset + k] = pos * omega[k];
                    }
                    writeOffset += axisPairs;
                }

                // ── 2. Build the head_dim-long freqs vector via stack-and-reshape:
                //    [a0, a0, a1, a1, ..., a_{halfDim-1}, a_{halfDim-1}].
                // Then store cos in [0, head_dim) and sin in [head_dim, 2*head_dim) of the output tensor. ──
                for (int k = 0; k < halfDim; k++)
                {
                    double angle = halfAngles[k];
                    float cv = (float)Math.Cos(angle);
                    float sv = (float)Math.Sin(angle);
                    int twoK = 2 * k;
                    outPtr[outBase + twoK + 0] = cv;
                    outPtr[outBase + twoK + 1] = cv;
                    outPtr[outBase + _headDim + twoK + 0] = sv;
                    outPtr[outBase + _headDim + twoK + 1] = sv;
                }
            }
        }

        return freqs;
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
