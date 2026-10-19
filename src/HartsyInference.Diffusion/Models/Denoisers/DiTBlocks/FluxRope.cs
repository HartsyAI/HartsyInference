using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Axial Rotary Position Embedding for Flux. Computes RoPE rotation matrices across 3 axes [16, 56, 56] with theta=10000, then applies 2x2 rotation to each Q/K pair. Precomputes cos/sin tables per resolution for caching.</summary>
public sealed unsafe class FluxRope : IDisposable
{
    /// <summary>Q/K vectors rotated per range of the host fan-out: a vector is only a few hundred operations, so a
    /// range of them amortizes dispatch, and the ranges depend only on the vector count.</summary>
    private const long RowsPerRange = 1024;

    private readonly RopeFrequencyPlan _plan;
    private readonly int _headDim;

    // Cached cos/sin tables: [totalSeqLen, headDim/2] interleaved
    private float[]? _cosCache;
    private float[]? _sinCache;
    private int _cachedSeqLen;

    // Device duplicated-pair tables [S, headDim] live in a per-instance RopeTableCache, one entry per backend. Two
    // IBackend instances (CFG-branch parallelism, or DiT block sharding) can use one FluxRope, so the cache keys by
    // backend and serializes its own state. Precompute retires the previous signature's tables before the new ones are
    // built, which also gives the owning transformer its step-graph invalidation through beforeRetire.
    private readonly object _gpuLock = new();
    private readonly RopeTableCache _deviceTables;
    private float[]? _positions;

    private static readonly RopeTableSettings DeviceTableSettings = new()
    {
        AngleOrder = RopeAngleOrder.MultiplyByInverse,
        Layout = RopeTableLayout.InterleavedPair,
        Preload = true,
        RetainedSignatures = 1,
    };

    /// <summary>Creates a FluxRope with the given per-axis dimensions.</summary>
    /// <param name="axesDim">Dimensions per axis. Default: [16, 56, 56]. Must be positive and even; the sum is the head dimension.</param>
    /// <param name="theta">Base frequency. Default: 10000.</param>
    /// <param name="beforeRetire">Runs for the owning backend before its device tables are freed. A transformer passes its
    /// step-graph invalidation here.</param>
    public FluxRope(int[]? axesDim = null, int theta = 10000, Action<IBackend>? beforeRetire = null)
    {
        int[] dims = axesDim ?? [16, 56, 56];
        _plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = dims, Theta = theta });
        _headDim = _plan.HeadDim;
        _deviceTables = new RopeTableCache(_plan, DeviceTableSettings, beforeRetire);
    }

    /// <summary>Total head dimension (sum of all axes dimensions, typically 128).</summary>
    public int HeadDim => _headDim;

    /// <summary>Precomputes cos/sin tables for the given position IDs. Call before Forward.</summary>
    /// <param name="posIds">Position IDs [totalSeqLen, numAxes] as float32. Text tokens: all zeros. Image tokens: [0, row, col].</param>
    public void Precompute(Tensor posIds)
    {
        int totalSeqLen = (int)posIds.Shape[0];
        int numAxes = (int)posIds.Shape[1];
        if (numAxes != _plan.Axes)
            throw new ArgumentException($"FluxRope is configured for {_plan.Axes} axes but the position IDs carry {numAxes}.", nameof(posIds));

        // Locked: two backends sharing this FluxRope (CFG-branch parallelism) can call Precompute concurrently for the same
        // step. Callers with DIFFERING signatures (e.g. cond/uncond token counts differ under regional conditioning) must not
        // be routed here concurrently at all; see FluxPipeline's cfgParallelEligible guard.
        lock (_gpuLock)
        {
            float[] cos = new float[totalSeqLen * (_headDim / 2)];
            float[] sin = new float[cos.Length];
            ReadOnlySpan<float> positions = new((void*)posIds.DataPointer, totalSeqLen * numAxes);
            MultiAxisRopeTables.ComputePairs(_plan, positions, totalSeqLen, RopeAngleOrder.MultiplyByInverse, cos, sin);

            _cosCache = cos;
            _sinCache = sin;
            _cachedSeqLen = totalSeqLen;
            _positions = positions.ToArray();
            _deviceTables.Clear();   // host tables changed → every backend's device tables are rebuilt on next use
        }
    }

    /// <summary>GPU-resident RoPE on a <c>[1, S, H, D]</c> (pre-permute) Q/K pair via
    /// <see cref="IBackend.WanRopeInterleaved"/> — the same interleaved-pair rotation as <see cref="Forward"/>
    /// (<c>x0' = c·x0 − s·x1, x1' = s·x0 + c·x1</c>), so results are bit-identical; text tokens (position ids all
    /// zero → cos 1 / sin 0) rotate by identity exactly as on the host path. Applied BEFORE the head permute:
    /// the pre-permute <c>[1, S, H, D]</c> layout is exactly WanRopeInterleaved's <c>[S, heads·headDim]</c>
    /// contract. Replaces the per-block host loop (full Q/K D2H→trig→H2D round trip per block per step).
    /// Requires batch 1 (flat-layout equivalence); callers keep the host <see cref="Forward"/> for B&gt;1.
    /// Mirrors <see cref="HunyuanImageRope.ApplyGpu"/>.</summary>
    public void ApplyGpu(IBackend backend, Tensor q, Tensor k, int numHeads)
    {
        CheckGpuPrecondition("ApplyGpu", q);
        using RopeTableLease lease = _deviceTables.Acquire(backend, 0, _cachedSeqLen, _positions!);
        (Tensor cos, Tensor sin) = DeviceTablesOf(lease);
        backend.WanRopeInterleaved(q, cos, sin, _cachedSeqLen, numHeads, _headDim);
        backend.WanRopeInterleaved(k, cos, sin, _cachedSeqLen, numHeads, _headDim);
    }

    /// <summary>GQA variant of <see cref="ApplyGpu"/>: rotates Q with <paramref name="numHeads"/> and K with
    /// <paramref name="numKvHeads"/>. Requires batch 1.</summary>
    public void ApplyGpuGqa(IBackend backend, Tensor q, Tensor k, int numHeads, int numKvHeads)
    {
        CheckGpuPrecondition("ApplyGpuGqa", q);
        using RopeTableLease lease = _deviceTables.Acquire(backend, 0, _cachedSeqLen, _positions!);
        (Tensor cos, Tensor sin) = DeviceTablesOf(lease);
        backend.WanRopeInterleaved(q, cos, sin, _cachedSeqLen, numHeads, _headDim);
        backend.WanRopeInterleaved(k, cos, sin, _cachedSeqLen, numKvHeads, _headDim);
    }

    /// <summary>Releases the device tables. Outstanding leases keep theirs until released.</summary>
    public void Dispose() => _deviceTables.Dispose();

    private void CheckGpuPrecondition(string method, Tensor q)
    {
        if (_cosCache == null || _sinCache == null)
            throw new InvalidOperationException($"FluxRope.Precompute must be called before {method}.");
        int seqLen = (int)q.Shape[1];
        if (seqLen != _cachedSeqLen)
            throw new InvalidOperationException($"FluxRope.{method}: tensor seqLen {seqLen} != precomputed {_cachedSeqLen}.");
    }

    private static (Tensor Cos, Tensor Sin) DeviceTablesOf(RopeTableLease lease) =>
        lease.DeviceCos is { } cos && lease.DeviceSin is { } sin
            ? (cos, sin)
            : throw new InvalidOperationException("FluxRope device tables require a preloading cache.");

    /// <summary>Applies RoPE rotation to Q and K tensors in-place. Q/K must be [B, numHeads, seqLen, headDim] laid out as contiguous floats. Precompute must be called first with matching seqLen.</summary>
    /// <param name="q">Query tensor [B, numHeads, seqLen, headDim]. Modified in-place.</param>
    /// <param name="k">Key tensor [B, numHeads, seqLen, headDim]. Modified in-place.</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="seqLen">Sequence length (must match precomputed length).</param>
    public void Forward(Tensor q, Tensor k, int batch, int numHeads, int seqLen)
    {
        if (_cosCache == null || _sinCache == null)
            throw new InvalidOperationException("FluxRope.Precompute must be called before Forward.");

        int halfDim = _headDim / 2;
        // q/k are [B, numHeads, seqLen, headDim]; each (b,h,s) vector rotates independently → parallelize over
        // the flattened b*h*s space. Single-threaded this loop dominates DiT inference (billions of element-ops
        // per forward across all blocks); identical math, so verified models (Flux/Ideogram) stay bit-exact.
        nint qBase = (nint)q.DataPointer, kBase = (nint)k.DataPointer;
        long rows = (long)batch * numHeads * seqLen;
        fixed (float* cosPtr = _cosCache, sinPtr = _sinCache)
        {
            nint cosBase = (nint)cosPtr, sinBase = (nint)sinPtr;
            CpuParallel.ForRanges(rows, RowsPerRange, 12L * halfDim, (start, length) =>
            {
                for (long row = start; row < start + length; row++)
                {
                    long s = row % seqLen;
                    int vecOffset = (int)(row * _headDim);
                    int freqIdx = (int)(s * halfDim);
                    ApplyRotation((float*)qBase + vecOffset, (float*)cosBase + freqIdx, (float*)sinBase + freqIdx, halfDim);
                    ApplyRotation((float*)kBase + vecOffset, (float*)cosBase + freqIdx, (float*)sinBase + freqIdx, halfDim);
                }
            });
        }
    }

    /// <summary>Applies RoPE rotation to a single Q or K tensor in-place. Use this for grouped-query attention, where
    /// Q and K have different head counts (so the two-tensor <see cref="Forward"/> can't rotate both at once). Tensor
    /// must be <c>[B, numHeads, seqLen, headDim]</c>; <see cref="Precompute"/> must have been called for this seqLen.</summary>
    public void ForwardSingle(Tensor qOrK, int batch, int numHeads, int seqLen)
    {
        if (_cosCache == null || _sinCache == null)
            throw new InvalidOperationException("FluxRope.Precompute must be called before ForwardSingle.");

        int halfDim = _headDim / 2;
        nint ptrBase = (nint)qOrK.DataPointer;
        long rows = (long)batch * numHeads * seqLen;
        fixed (float* cosPtr = _cosCache, sinPtr = _sinCache)
        {
            nint cosBase = (nint)cosPtr, sinBase = (nint)sinPtr;
            CpuParallel.ForRanges(rows, RowsPerRange, 6L * halfDim, (start, length) =>
            {
                for (long row = start; row < start + length; row++)
                {
                    long s = row % seqLen;
                    int vecOffset = (int)(row * _headDim);
                    ApplyRotation((float*)ptrBase + vecOffset, (float*)cosBase + (int)(s * halfDim), (float*)sinBase + (int)(s * halfDim), halfDim);
                }
            });
        }
    }

    /// <summary>Applies 2x2 rotation to each adjacent pair of elements in a vector. x[2i]' = cos * x[2i] - sin * x[2i+1], x[2i+1]' = sin * x[2i] + cos * x[2i+1].</summary>
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

    /// <summary>Builds position IDs for Flux: text tokens get [0,0,0], image tokens get [0,row,col]. Returns a Tensor of shape [txtSeqLen + imgSeqLen, 3].</summary>
    /// <param name="txtSeqLen">Number of text tokens.</param>
    /// <param name="hPacked">Packed image height (latent_h / 2).</param>
    /// <param name="wPacked">Packed image width (latent_w / 2).</param>
    public static Tensor BuildPositionIds(int txtSeqLen, int hPacked, int wPacked)
    {
        int imgSeqLen = hPacked * wPacked;
        int totalSeqLen = txtSeqLen + imgSeqLen;
        TensorShape shape = new TensorShape(totalSeqLen, 3);
        Tensor posIds = new Tensor(shape, DType.F32);

        float* ptr = (float*)posIds.DataPointer;

        // Text tokens: all zeros (no spatial position)
        for (int i = 0; i < txtSeqLen * 3; i++)
            ptr[i] = 0f;

        // Image tokens: [0, row, col]
        for (int row = 0; row < hPacked; row++)
        {
            for (int col = 0; col < wPacked; col++)
            {
                int idx = txtSeqLen + row * wPacked + col;
                ptr[idx * 3 + 0] = 0f;
                ptr[idx * 3 + 1] = row;
                ptr[idx * 3 + 2] = col;
            }
        }

        return posIds;
    }

    /// <summary>Flux.1 Kontext position ids for the sequence <c>[text ; noise-image ; reference-image]</c>.
    /// Matches diffusers <c>FluxKontextPipeline</c>: text tokens are all-zero; the noise image grid is
    /// <c>[0, row, col]</c>; the reference image grid uses the SAME spatial (row, col) but sets the first
    /// (temporal) axis to <c>1</c> so RoPE distinguishes reference tokens from co-located noise tokens
    /// (<c>image_ids[..., 0] = 1</c>).</summary>
    public static Tensor BuildPositionIdsKontext(int txtSeqLen, int hPacked, int wPacked, int refHPacked, int refWPacked)
    {
        int noiseSeqLen = hPacked * wPacked;
        int refSeqLen = refHPacked * refWPacked;
        int totalSeqLen = txtSeqLen + noiseSeqLen + refSeqLen;
        Tensor posIds = new Tensor(new TensorShape(totalSeqLen, 3), DType.F32);
        float* ptr = (float*)posIds.DataPointer;

        for (int i = 0; i < txtSeqLen * 3; i++) ptr[i] = 0f;

        for (int row = 0; row < hPacked; row++)
            for (int col = 0; col < wPacked; col++)
            {
                int idx = txtSeqLen + row * wPacked + col;
                ptr[idx * 3 + 0] = 0f;
                ptr[idx * 3 + 1] = row;
                ptr[idx * 3 + 2] = col;
            }

        for (int row = 0; row < refHPacked; row++)
            for (int col = 0; col < refWPacked; col++)
            {
                int idx = txtSeqLen + noiseSeqLen + row * refWPacked + col;
                ptr[idx * 3 + 0] = 1f;  // reference: temporal axis = 1
                ptr[idx * 3 + 1] = row;
                ptr[idx * 3 + 2] = col;
            }

        return posIds;
    }
}
