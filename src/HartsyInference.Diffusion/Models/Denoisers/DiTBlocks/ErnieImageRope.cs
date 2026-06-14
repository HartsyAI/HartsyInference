using System.Runtime.CompilerServices;
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

    /// <summary>Computes the cos/sin freqs tensor used by <see cref="ApplyRotaryEmb"/>. Output shape is <c>[B, totalSeq, head_dim]</c> packed as <c>[cos_0..cos_{D-1}, sin_0..sin_{D-1}]</c> per token along the last dim — but stored as two separate halves of length <c>head_dim</c>: a cos block then a sin block, total length <c>2 * head_dim</c>. We pack both halves into one tensor so the GPU upload path is a single buffer; <see cref="ApplyRotaryEmb"/> indexes them as <c>cos = freqs[..., :head_dim]</c> and <c>sin = freqs[..., head_dim:]</c>.</summary>
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

    /// <summary>Applies non-interleaved Megatron-style rotation to Q and K in-place.
    /// <para>Layout requirements:
    /// <list type="bullet">
    ///   <item><paramref name="q"/>, <paramref name="k"/> shape: <c>[B, S, numHeads, headDim]</c> (heads-as-3rd-dim layout — matches <c>transformer_ernie_image.py</c>'s <c>unflatten(-1, (heads, -1))</c> step).</item>
    ///   <item><paramref name="freqs"/> shape: <c>[B, S, 2 * headDim]</c> (from <see cref="BuildFreqs"/>): cos in <c>[..., :headDim]</c>, sin in <c>[..., headDim:]</c>.</item>
    /// </list></para>
    /// <para>The rotation is <c>out = x * cos + rotate_half(x) * sin</c> with <c>rotate_half(x) = cat(-x[..., halfDim:], x[..., :halfDim])</c>.</para>
    /// </summary>
    public void ApplyRotaryEmb(Tensor q, Tensor k, Tensor freqs)
    {
        int batch = (int)q.Shape[0];
        int seqLen = (int)q.Shape[1];
        int numHeads = (int)q.Shape[2];
        int headDim = (int)q.Shape[3];
        if (headDim != _headDim)
            throw new ArgumentException($"Q head_dim {headDim} != rope head_dim {_headDim}.", nameof(q));
        if (k.Shape[0] != batch || k.Shape[1] != seqLen || k.Shape[2] != numHeads || k.Shape[3] != headDim)
            throw new ArgumentException("K shape must match Q shape.", nameof(k));
        if (freqs.Shape[0] != batch || freqs.Shape[1] != seqLen || freqs.Shape[2] != 2 * headDim)
            throw new ArgumentException($"freqs shape must be [{batch}, {seqLen}, {2 * headDim}].", nameof(freqs));

        int halfDim = headDim / 2;
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;
        float* fPtr = (float*)freqs.DataPointer;

        // Single-vector scratch reused across every (b,s,h) — sized for the maximum halfDim we ever see.
        // halfDim = 64 for the default (32,48,48) → 512 bytes. Allocate once at the call site to avoid
        // hammering the stack from a deep tight loop.
        Span<float> lowerSnapshot = stackalloc float[halfDim];

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                long freqBase = ((long)b * seqLen + s) * (2L * headDim);
                float* cosVec = fPtr + freqBase;
                float* sinVec = fPtr + freqBase + headDim;

                for (int h = 0; h < numHeads; h++)
                {
                    long vecOff = (((long)b * seqLen + s) * numHeads + h) * headDim;
                    ApplyNonInterleavedRotation(qPtr + vecOff, cosVec, sinVec, halfDim, lowerSnapshot);
                    ApplyNonInterleavedRotation(kPtr + vecOff, cosVec, sinVec, halfDim, lowerSnapshot);
                }
            }
        }
    }

    /// <summary>Non-interleaved rotation: pairs are <c>(i, i + halfDim)</c>. Computes <c>out[i] = x[i]*cos[i] - x[i+halfDim]*sin[i]</c> for <c>i &lt; halfDim</c> and <c>out[i] = x[i]*cos[i] + x[i-halfDim]*sin[i]</c> for <c>i &gt;= halfDim</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyNonInterleavedRotation(float* vec, float* cos, float* sin, int halfDim, Span<float> lowerSnapshot)
    {
        // Snapshot the lower half before overwriting — the upper-half write reads original lower-half values.
        for (int i = 0; i < halfDim; i++)
            lowerSnapshot[i] = vec[i];

        for (int i = 0; i < halfDim; i++)
        {
            float lower = lowerSnapshot[i];
            float upper = vec[i + halfDim];
            vec[i] = lower * cos[i] - upper * sin[i];
            vec[i + halfDim] = upper * cos[i + halfDim] + lower * sin[i + halfDim];
        }
    }
}
