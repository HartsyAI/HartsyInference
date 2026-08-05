using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>2-axis rotary position embedding for Hunyuan Image 2.1 (<c>HunyuanImageRotaryPosEmbed</c>). Splits the per-head dimension into <c>(height, width)</c> sub-bands (default <c>[64, 64]</c>) and rotates each pair of features by <c>theta = position * (1 / base^(2k/axisDim))</c>. Applied to image tokens only — text tokens are kept unrotated and concatenated after rotation. Each token at packed-grid <c>(row, col)</c> uses position <c>[row, col]</c>; both axes contribute <c>axesDim/2</c> cos/sin pairs to the head vector. Diffusers uses <c>theta=256</c> by default and <c>get_1d_rotary_pos_embed</c> with <c>use_real=True</c> producing <c>(cos, sin)</c> tables that are concatenated along the feature axis before being applied as the standard <c>(real, imag)</c> 2x2 rotation; this implementation produces the same numerical output by composing the per-axis tables in the same axis order.</summary>
public sealed unsafe class HunyuanImageRope
{
    private readonly int[] _axesDim;
    private readonly float _theta;
    private readonly int _headDim;
    // Cached PER BACKEND (the QwenImageRope/FluxRope precedent): DiT sharding runs blocks [split, BlockCount) on
    // a second backend, and two backends staging the SAME host tensor through WanRopeInterleaved trips the
    // transfer helper's binding/pinning state (observed CUDA_ERROR_INVALID_VALUE on the second card at real
    // table sizes) — each backend gets its own host build, preloaded into that backend's own weight cache.
    private readonly Dictionary<IBackend, (int[] Dims, Tensor Cos, Tensor Sin)> _tables = new();

    /// <summary>Creates a HunyuanImageRope.</summary>
    /// <param name="axesDim">Per-axis dim split. 2 axes <c>(height, width)</c> for image (default <c>[64, 64]</c>);
    /// 3 axes <c>(temporal, height, width)</c> for video (e.g. <c>[16, 56, 56]</c> for HunyuanVideo). Each must be even.</param>
    /// <param name="theta">RoPE base frequency. Default 256.</param>
    public HunyuanImageRope(int[]? axesDim = null, float theta = 256.0f)
    {
        _axesDim = axesDim ?? [64, 64];
        if (_axesDim.Length < 1)
            throw new ArgumentException("HunyuanImageRope requires at least 1 axis.", nameof(axesDim));
        _theta = theta;
        _headDim = 0;
        for (int i = 0; i < _axesDim.Length; i++)
            _headDim += _axesDim[i];
    }

    /// <summary>Sum of per-axis dimensions; equals the per-head dim of the wrapped attention block.</summary>
    public int HeadDim => _headDim;

    /// <summary>Per-axis dim split <c>(height, width)</c>. Each must be even — pairs of features get rotated together.</summary>
    public ReadOnlySpan<int> AxesDim => _axesDim;

    /// <summary>Rotates Q and K in-place for image tokens only. Tokens are laid out row-major over <paramref name="hPacked"/> × <paramref name="wPacked"/>; token at index <c>r * W + c</c> uses position <c>(r, c)</c>. Both Q and K must be <c>[B, numHeads, imgSeqLen, headDim]</c> and <c>imgSeqLen == hPacked * wPacked</c>.</summary>
    public void ApplyImage(Tensor q, Tensor k, int batch, int numHeads, int hPacked, int wPacked)
    {
        Span<int> dims = stackalloc int[2] { hPacked, wPacked };
        ApplyJoint(q, k, batch, numHeads, dims);
    }

    /// <summary>Rotates Q and K in-place for image/video tokens laid out row-major over
    /// <paramref name="packedDims"/> (e.g. <c>[h, w]</c> for image or <c>[t, h, w]</c> for video). The number
    /// of dims must equal the axis count the rope was constructed with, and their product must equal the token
    /// sequence length. Q and K are <c>[B, numHeads, seqLen, headDim]</c>. Text tokens are roped separately
    /// (identity) by the caller — this rotates only the passed image/video tokens.</summary>
    public void ApplyJoint(Tensor q, Tensor k, int batch, int numHeads, ReadOnlySpan<int> packedDims)
    {
        if (packedDims.Length != _axesDim.Length)
            throw new ArgumentException($"packedDims has {packedDims.Length} dims but rope has {_axesDim.Length} axes.", nameof(packedDims));
        int seqLen = 1;
        for (int i = 0; i < packedDims.Length; i++) seqLen *= packedDims[i];
        int halfDim = _headDim / 2;
        float[] cosTable = new float[seqLen * halfDim];
        float[] sinTable = new float[seqLen * halfDim];

        Span<double> pos = stackalloc double[packedDims.Length];
        for (int s = 0; s < seqLen; s++)
        {
            // Decode row-major coordinates for token s over packedDims.
            int rem = s;
            for (int axis = packedDims.Length - 1; axis >= 0; axis--)
            {
                pos[axis] = rem % packedDims[axis];
                rem /= packedDims[axis];
            }
            FillTokenFreqs(cosTable, sinTable, s, pos);
        }

        ApplyRotationBatched(q, k, cosTable, sinTable, batch, numHeads, seqLen);
    }

    /// <summary>GPU-resident RoPE on a <c>[1, S, H, D]</c> Q/K pair via <see cref="IBackend.WanRopeInterleaved"/>
    /// (same interleaved-pair rotation): rotates the leading <c>Π packedDims</c> tokens in place, leaving any trailing
    /// (text) rows untouched — so it works both on a standalone image tensor and on a joint <c>[img, txt]</c> tensor
    /// with image-first layout. Replaces the per-block host path (a full Q/K D2H→trig→H2D round trip per block per
    /// step); the duplicated-pair cos/sin tables <c>[S, headDim]</c> are built ONCE per packed-dims and preloaded to
    /// the GPU weight cache. Requires batch 1 (the flat layout equivalence); callers keep the host path for B&gt;1.</summary>
    public void ApplyGpu(IBackend backend, Tensor q, Tensor k, int numHeads, ReadOnlySpan<int> packedDims)
    {
        (Tensor cos, Tensor sin, int seqLen) = GetTables(backend, packedDims);
        backend.WanRopeInterleaved(q, cos, sin, seqLen, numHeads, _headDim);
        backend.WanRopeInterleaved(k, cos, sin, seqLen, numHeads, _headDim);
    }

    /// <summary>Builds (or returns the cached) duplicated-pair cos/sin tables <c>[S, headDim]</c> for
    /// <paramref name="packedDims"/> — the value for pair <c>i</c> stored at both <c>2i</c> and <c>2i+1</c>, the
    /// layout <see cref="IBackend.WanRopeInterleaved"/> reads. One table set per backend (see the field note),
    /// preloaded into that backend's weight cache so the per-block op skips the H2D upload; frees that backend's
    /// previous tables when the grid changes.</summary>
    private (Tensor cos, Tensor sin, int seqLen) GetTables(IBackend backend, ReadOnlySpan<int> packedDims)
    {
        if (packedDims.Length != _axesDim.Length)
            throw new ArgumentException($"packedDims has {packedDims.Length} dims but rope has {_axesDim.Length} axes.", nameof(packedDims));
        int seqLen = 1;
        for (int i = 0; i < packedDims.Length; i++) seqLen *= packedDims[i];

        bool hasEntry = _tables.TryGetValue(backend, out (int[] Dims, Tensor Cos, Tensor Sin) entry);
        if (hasEntry && packedDims.SequenceEqual(entry.Dims))
            return (entry.Cos, entry.Sin, seqLen);

        if (hasEntry)
        {
            backend.FreeWeights([entry.Cos, entry.Sin]);
            entry.Cos.Dispose();
            entry.Sin.Dispose();
        }

        int halfDim = _headDim / 2;
        float[] compactCos = new float[halfDim];
        float[] compactSin = new float[halfDim];
        Tensor cos = new Tensor(new TensorShape(seqLen, _headDim), DType.F32);
        Tensor sin = new Tensor(new TensorShape(seqLen, _headDim), DType.F32);
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;
        Span<double> pos = stackalloc double[packedDims.Length];
        for (int s = 0; s < seqLen; s++)
        {
            int rem = s;
            for (int axis = packedDims.Length - 1; axis >= 0; axis--)
            {
                pos[axis] = rem % packedDims[axis];
                rem /= packedDims[axis];
            }
            FillTokenFreqs(compactCos, compactSin, 0, pos);
            long baseOff = (long)s * _headDim;
            for (int i = 0; i < halfDim; i++)
            {
                cp[baseOff + 2 * i] = compactCos[i]; cp[baseOff + 2 * i + 1] = compactCos[i];
                sp[baseOff + 2 * i] = compactSin[i]; sp[baseOff + 2 * i + 1] = compactSin[i];
            }
        }

        backend.PreloadWeights([cos, sin]);
        _tables[backend] = (packedDims.ToArray(), cos, sin);
        return (cos, sin, seqLen);
    }

    private void FillTokenFreqs(Span<float> cosTable, Span<float> sinTable, int seqIdx, ReadOnlySpan<double> positions)
    {
        int halfDim = _headDim / 2;
        int destOffset = seqIdx * halfDim;
        int freqOffset = 0;

        for (int axis = 0; axis < _axesDim.Length; axis++)
        {
            int axisDim = _axesDim[axis];
            int numPairs = axisDim / 2;
            double pos = positions[axis];

            for (int kIdx = 0; kIdx < numPairs; kIdx++)
            {
                double inv = 1.0 / Math.Pow(_theta, (double)(2 * kIdx) / axisDim);
                double angle = pos * inv;
                cosTable[destOffset + freqOffset + kIdx] = (float)Math.Cos(angle);
                sinTable[destOffset + freqOffset + kIdx] = (float)Math.Sin(angle);
            }
            freqOffset += numPairs;
        }
    }

    private void ApplyRotationBatched(Tensor q, Tensor k, ReadOnlySpan<float> cosTable, ReadOnlySpan<float> sinTable,
        int batch, int numHeads, int seqLen)
    {
        int halfDim = _headDim / 2;
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;

        fixed (float* cosPtr = cosTable, sinPtr = sinTable)
        {
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        int vecOffset = ((b * numHeads + h) * seqLen + s) * _headDim;
                        int freqIdx = s * halfDim;
                        ApplyRotation(qPtr + vecOffset, cosPtr + freqIdx, sinPtr + freqIdx, halfDim);
                        ApplyRotation(kPtr + vecOffset, cosPtr + freqIdx, sinPtr + freqIdx, halfDim);
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyRotation(float* vec, float* cos, float* sin, int halfDim)
    {
        for (int i = 0; i < halfDim; i++)
        {
            float x0 = vec[2 * i];
            float x1 = vec[2 * i + 1];
            vec[2 * i] = cos[i] * x0 - sin[i] * x1;
            vec[2 * i + 1] = sin[i] * x0 + cos[i] * x1;
        }
    }
}
