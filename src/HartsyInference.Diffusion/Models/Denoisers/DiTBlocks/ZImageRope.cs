using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Multi-axis rotary position embedding for Z-Image (Lumina2/NextDiT). Splits the head dimension across 3 axes [frame, height, width] = [32, 48, 48] with θ=256. Caption tokens get position [0,0,0]; image tokens get [0, row, col]. Caption pad tokens and image pad tokens use the same axis-0=0 fallback so RoPE is well-defined for the entire concatenated sequence.</summary>
/// <remarks>Host pairs come from <see cref="MultiAxisRopeTables"/>. The device rows are duplicated pairs in a
/// <see cref="RopeTableCache"/>; the kernel reads only the even entries, and the odd entries are duplicates rather than zeros.</remarks>
public sealed unsafe class ZImageRope : IQkRotation, IHeadMajorQkRotation
{
    private static readonly RopeTableSettings DeviceTableSettings = new()
    {
        AngleOrder = RopeAngleOrder.MultiplyByInverse,
        Layout = RopeTableLayout.InterleavedPair,
        Preload = true,
        RetainedSignatures = 1,
    };

    private readonly RopeFrequencyPlan _plan;
    private readonly int _headDim;
    private readonly RopeTableCache _deviceTables;

    private float[]? _cosCache;
    private float[]? _sinCache;
    private float[]? _positions;
    private int _cachedSeqLen;

    /// <summary>Creates a Z-Image RoPE with the given axis dims and base.</summary>
    /// <param name="axesDim">Per-axis dimensions. Must sum to head_dim. Default [32, 48, 48] (sums to 128).</param>
    /// <param name="theta">Base frequency. Default 256 (Z-Image; much smaller than Flux's 10000).</param>
    public ZImageRope(int[]? axesDim = null, float theta = 256.0f)
    {
        _plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = axesDim ?? [32, 48, 48], Theta = theta });
        _headDim = _plan.HeadDim;
        _deviceTables = new RopeTableCache(_plan, DeviceTableSettings);
    }

    /// <summary>Total head dimension (sum of all axes, 128 for Z-Image).</summary>
    public int HeadDim => _headDim;

    /// <summary>Precomputes per-token cos/sin tables across all 3 axes for the given position IDs. Call before <see cref="Forward"/>.</summary>
    /// <param name="posIds">Position IDs [totalSeqLen, numAxes] as F32. For Z-Image, axes are [frame=0, row, col] for image tokens; all-zero for caption + pad tokens.</param>
    public void Precompute(Tensor posIds)
    {
        int totalSeqLen = (int)posIds.Shape[0];
        int numAxes = (int)posIds.Shape[1];
        if (numAxes != _plan.Axes)
            throw new ArgumentException($"ZImageRope is configured for {_plan.Axes} axes but the position IDs carry {numAxes}.", nameof(posIds));

        ReadOnlySpan<float> positions = new((void*)posIds.DataPointer, totalSeqLen * numAxes);
        float[] cos = new float[totalSeqLen * (_headDim / 2)];
        float[] sin = new float[cos.Length];
        MultiAxisRopeTables.ComputePairs(_plan, positions, totalSeqLen, RopeAngleOrder.MultiplyByInverse, cos, sin);

        _cosCache = cos;
        _sinCache = sin;
        _positions = positions.ToArray();
        _cachedSeqLen = totalSeqLen;
        _deviceTables.Clear();   // new positions → every backend's device tables are rebuilt on next use
    }

    /// <summary>Applies RoPE rotation to Q/K in-place when both share the same head count (full MHA). Q/K shape [B, numHeads, seqLen, headDim]. <see cref="Precompute"/> must have been called with matching seqLen.</summary>
    public void Forward(Tensor q, Tensor k, int batch, int numHeads, int seqLen)
    {
        ForwardSingle(q, batch, numHeads, seqLen);
        ForwardSingle(k, batch, numHeads, seqLen);
    }

    /// <summary>Applies RoPE rotation to a single tensor in-place. Tensor shape [B, numHeads, seqLen, headDim]. Used when Q and K have different head counts (GQA — call once per tensor with its own head count).</summary>
    public void ForwardSingle(Tensor t, int batch, int numHeads, int seqLen)
    {
        if (_cosCache == null || _sinCache == null)
            throw new InvalidOperationException("ZImageRope.Precompute must be called before Forward.");

        int halfDim = _headDim / 2;
        float* tPtr = (float*)t.DataPointer;

        fixed (float* cosPtr = _cosCache, sinPtr = _sinCache)
        {
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        int vecOffset = ((b * numHeads + h) * seqLen + s) * _headDim;
                        int freqIdx = s * halfDim;
                        ApplyRotation(tPtr + vecOffset, cosPtr + freqIdx, sinPtr + freqIdx, halfDim);
                    }
                }
            }
        }
    }

    /// <summary>Device RoPE (B=1, the pipeline case): rotates Q and K in-place on the PRE-permute
    /// <c>[1, seqLen, numHeads, headDim]</c> layout via <see cref="IBackend.WanRopeInterleaved"/> — the same
    /// interleaved-pair rotation as <see cref="ApplyRotation"/> (adjacent pair (2i, 2i+1), angle shared across
    /// heads), bit-identical to the host path because RoPE indexes only (position, dim). This replaces the host
    /// <see cref="Forward"/> that D2H-round-tripped the full ~63 MB Q and K activations every block every step
    /// (the dominant residual Z-Image cost after the block GPU-residency rewrite). Mirrors
    /// <c>FluxRope.ApplyGpuGqa</c> (the Krea2 port). B&gt;1 callers keep the host <see cref="Forward"/>.</summary>
    public void ApplyGpu(IBackend backend, Tensor q, Tensor k, int numHeads)
    {
        int seqLen = CheckGpuPrecondition("ApplyGpu", q);
        using RopeTableLease lease = _deviceTables.Acquire(backend, 0, seqLen, _positions!);
        (Tensor cos, Tensor sin) = DeviceTablesOf(lease);
        backend.WanRopeInterleaved(q, cos, sin, seqLen, numHeads, _headDim);
        backend.WanRopeInterleaved(k, cos, sin, seqLen, numHeads, _headDim);
    }

    /// <summary>GQA variant of <see cref="ApplyGpu"/>: rotates Q with <paramref name="numHeads"/> and K with
    /// <paramref name="numKvHeads"/> (RoPE indexes only (position, dim) — head count is just layout), same
    /// pre-permute <c>[1, S, H, D]</c> contract. Used by the Lumina2 GPU-resident block path.</summary>
    public void ApplyGpuGqa(IBackend backend, Tensor q, Tensor k, int numHeads, int numKvHeads)
    {
        int seqLen = CheckGpuPrecondition("ApplyGpuGqa", q);
        using RopeTableLease lease = _deviceTables.Acquire(backend, 0, seqLen, _positions!);
        (Tensor cos, Tensor sin) = DeviceTablesOf(lease);
        backend.WanRopeInterleaved(q, cos, sin, seqLen, numHeads, _headDim);
        backend.WanRopeInterleaved(k, cos, sin, seqLen, numKvHeads, _headDim);
    }

    /// <summary>Releases the device-resident table copies for <paramref name="backend"/> while preserving the CPU precompute.
    /// The next GPU application re-uploads the current tables. This is required at a model-phase boundary because the tables
    /// are explicitly preloaded weights and therefore are not part of the transformer's checkpoint enumeration. A table that
    /// is still leased is freed by its last release.</summary>
    public void ReleaseGpuTables(IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        _deviceTables.ReleaseBackend(backend);
    }

    private int CheckGpuPrecondition(string method, Tensor q)
    {
        if (_cosCache == null || _sinCache == null || _positions == null)
            throw new InvalidOperationException($"ZImageRope.Precompute must be called before {method}.");
        int seqLen = (int)q.Shape[1];
        if (seqLen != _cachedSeqLen)
            throw new InvalidOperationException($"ZImageRope.{method}: tensor seqLen {seqLen} != precomputed {_cachedSeqLen}.");
        return seqLen;
    }

    private static (Tensor Cos, Tensor Sin) DeviceTablesOf(RopeTableLease lease) =>
        lease.DeviceCos is { } cos && lease.DeviceSin is { } sin
            ? (cos, sin)
            : throw new InvalidOperationException("Z-Image RoPE device tables require a preloading cache.");

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

    /// <summary>Builds position IDs for Z-Image's main DiT in [image, caption] concat order. See <see cref="ZImagePosEmbed"/>.</summary>
    public static Tensor BuildPositionIds(int capPaddedLen, int hPacked, int wPacked, int imgPaddedLen) =>
        ZImagePosEmbed.BuildPositionIds(capPaddedLen, hPacked, wPacked, imgPaddedLen);

    /// <summary>Builds caption-only position IDs for the context_refiner stack. See <see cref="ZImagePosEmbed"/>.</summary>
    public static Tensor BuildCaptionPositionIds(int capPaddedLen) => ZImagePosEmbed.BuildCaptionPositionIds(capPaddedLen);

    /// <summary>Builds image-only position IDs for the noise_refiner stack. See <see cref="ZImagePosEmbed"/>.</summary>
    public static Tensor BuildImagePositionIds(int hPacked, int wPacked, int imgPaddedLen) =>
        ZImagePosEmbed.BuildImagePositionIds(hPacked, wPacked, imgPaddedLen);

    /// <inheritdoc/>
    void IQkRotation.RotateTokenMajor(IBackend backend, Tensor q, Tensor k, int heads, int kvHeads) =>
        ApplyGpuGqa(backend, q, k, heads, kvHeads);

    /// <inheritdoc/>
    /// <remarks>Equal head counts only, as Z-Image's batched path runs.</remarks>
    void IHeadMajorQkRotation.RotateHeadMajor(IBackend backend, Tensor q, Tensor k, int batch, int heads, int kvHeads, int tokens)
    {
        if (kvHeads != heads)
            throw new NotSupportedException("ZImageRope rotates head-major Q and K with equal head counts only.");
        Forward(q, k, batch, heads, tokens);
    }
}
