using System.Collections.Generic;
using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Axial Rotary Position Embedding for Flux. Computes RoPE rotation matrices across 3 axes [16, 56, 56] with theta=10000, then applies 2x2 rotation to each Q/K pair. Precomputes cos/sin tables per resolution for caching.</summary>
public sealed unsafe class FluxRope
{
    private readonly int[] _axesDim;
    private readonly int _theta;
    private readonly int _headDim;

    // Cached cos/sin tables: [totalSeqLen, headDim/2] interleaved
    private float[]? _cosCache;
    private float[]? _sinCache;
    private int _cachedSeqLen;

    // GPU duplicated-pair tables [S, headDim] (pair i stored at both 2i and 2i+1 — the layout
    // IBackend.WanRopeInterleaved reads), one pair PER BACKEND: CFG-branch parallelism and DiT block
    // sharding both route two IBackend instances through one shared transformer/rope object (the
    // transformer's `_rope` field is not duplicated per backend), so ApplyGpu/ApplyGpuGqa can be called
    // for two different backends against the same FluxRope — concurrently (CFG-parallel: cond/uncond on
    // two threads) or sequentially (a block-range shard: backend A's block range, then backend B's).
    // WanRopeInterleaved re-stages cos/sin fresh via GpuTransferHelper.CopyToDevice/FreeDevice on every
    // call (not a weight-cache hit), so a single shared Tensor object used by two backends already reads
    // correctly regardless of which one built it — a stale reference here is not a wrong-device numerics
    // bug. The real hazard: a single unkeyed slot's rebuild-on-dirty path called backend.FreeWeights +
    // Tensor.Dispose() unconditionally, which can tear the tensor down while ANOTHER backend/thread is
    // mid-CopyToDevice against it (the same class of lifecycle race as the EnsureCpuData fix elsewhere in
    // this plan) — and concurrent Precompute calls raced unsynchronized writes to _cosCache/_sinCache.
    // _gpuLock serializes both; the per-backend keying also avoids redundant rebuild churn when two
    // backends alternate through one FluxRope.
    private readonly object _gpuLock = new();
    private readonly Dictionary<IBackend, (Tensor Cos, Tensor Sin)> _gpuTables = new();
    private bool _gpuTablesDirty = true;

    /// <summary>Creates a FluxRope with the given per-axis dimensions.</summary>
    /// <param name="axesDim">Dimensions per axis. Default: [16, 56, 56]. Must sum to headDim.</param>
    /// <param name="theta">Base frequency. Default: 10000.</param>
    public FluxRope(int[]? axesDim = null, int theta = 10000)
    {
        _axesDim = axesDim ?? [16, 56, 56];
        _theta = theta;
        _headDim = 0;
        for (int i = 0; i < _axesDim.Length; i++)
            _headDim += _axesDim[i];
    }

    /// <summary>Total head dimension (sum of all axes dimensions, typically 128).</summary>
    public int HeadDim => _headDim;

    /// <summary>Precomputes cos/sin tables for the given position IDs. Call before Forward.</summary>
    /// <param name="posIds">Position IDs [totalSeqLen, numAxes] as float32. Text tokens: all zeros. Image tokens: [0, row, col].</param>
    public void Precompute(Tensor posIds)
    {
        // Locked: two backends sharing this FluxRope (CFG-branch parallelism) can call Precompute
        // concurrently for the same step. Without this, concurrent writers race on _cosCache/_sinCache/
        // _cachedSeqLen (a torn read is possible even when both calls compute identical values, since
        // .NET gives no ordering guarantee between the array-content writes and the field publish without
        // a barrier). Callers with DIFFERING signatures (e.g. cond/uncond token counts differ under
        // regional conditioning) must not be routed here concurrently at all — one _cosCache instance
        // cannot correctly serve two different signatures at once; see FluxPipeline's cfgParallelEligible
        // guard, which requires matching signatures before allowing the concurrent branch.
        lock (_gpuLock)
        {
            int totalSeqLen = (int)posIds.Shape[0];
            int numAxes = (int)posIds.Shape[1];
            int halfDim = _headDim / 2;

            _cosCache = new float[totalSeqLen * halfDim];
            _sinCache = new float[totalSeqLen * halfDim];
            _cachedSeqLen = totalSeqLen;

            float* posPtr = (float*)posIds.DataPointer;
            int freqOffset = 0;

            // Max pairs across all axes (56/2=28 for Flux)
            int maxPairs = 0;
            for (int a = 0; a < numAxes; a++)
                maxPairs = Math.Max(maxPairs, _axesDim[a] / 2);
            Span<double> omega = stackalloc double[maxPairs];

            for (int axis = 0; axis < numAxes; axis++)
            {
                int axisDim = _axesDim[axis];
                int numPairs = axisDim / 2;

                // Precompute omega for this axis: omega[k] = 1 / (theta ^ (2k / axisDim))
                for (int k = 0; k < numPairs; k++)
                {
                    double scale = (double)(2 * k) / axisDim;
                    omega[k] = 1.0 / Math.Pow(_theta, scale);
                }

                for (int s = 0; s < totalSeqLen; s++)
                {
                    double pos = posPtr[s * numAxes + axis];

                    for (int k = 0; k < numPairs; k++)
                    {
                        double angle = pos * omega[k];
                        int idx = s * halfDim + freqOffset + k;
                        _cosCache[idx] = (float)Math.Cos(angle);
                        _sinCache[idx] = (float)Math.Sin(angle);
                    }
                }

                freqOffset += numPairs;
            }

            _gpuTablesDirty = true;   // host tables changed → every backend's GPU tables must be rebuilt
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
        if (_cosCache == null || _sinCache == null)
            throw new InvalidOperationException("FluxRope.Precompute must be called before ApplyGpu.");
        int seqLen = (int)q.Shape[1];
        if (seqLen != _cachedSeqLen)
            throw new InvalidOperationException($"FluxRope.ApplyGpu: tensor seqLen {seqLen} != precomputed {_cachedSeqLen}.");

        // DIAGNOSTIC (HARTSY_SKIP_ROPE=1): skip rope entirely (parity with Forward).
        if (Environment.GetEnvironmentVariable("HARTSY_SKIP_ROPE") == "1") return;

        (Tensor cos, Tensor sin) = GetGpuTables(backend);
        backend.WanRopeInterleaved(q, cos, sin, _cachedSeqLen, numHeads, _headDim);
        backend.WanRopeInterleaved(k, cos, sin, _cachedSeqLen, numHeads, _headDim);
    }

    /// <summary>GQA variant of <see cref="ApplyGpu"/>: rotates Q with <paramref name="numHeads"/> and K with
    /// <paramref name="numKvHeads"/> (grouped-query attention, where K has fewer heads than Q). Same pre-permute
    /// <c>[1, S, H, D]</c> contract and bit-identical interleaved-pair rotation as <see cref="ApplyGpu"/> —
    /// <see cref="IBackend.WanRopeInterleaved"/> is single-tensor and takes an explicit head count, so the only
    /// reason the shared-count <see cref="ApplyGpu"/> couldn't serve GQA was the hardcoded single count. RoPE is
    /// head-independent (cos/sin index position×dim, not head), so rotating Q and K with their own counts is exact.
    /// Requires batch 1; callers keep the host <see cref="ForwardSingle"/> for B&gt;1.</summary>
    public void ApplyGpuGqa(IBackend backend, Tensor q, Tensor k, int numHeads, int numKvHeads)
    {
        if (_cosCache == null || _sinCache == null)
            throw new InvalidOperationException("FluxRope.Precompute must be called before ApplyGpuGqa.");
        int seqLen = (int)q.Shape[1];
        if (seqLen != _cachedSeqLen)
            throw new InvalidOperationException($"FluxRope.ApplyGpuGqa: tensor seqLen {seqLen} != precomputed {_cachedSeqLen}.");

        // DIAGNOSTIC (HARTSY_SKIP_ROPE=1): skip rope entirely (parity with Forward).
        if (Environment.GetEnvironmentVariable("HARTSY_SKIP_ROPE") == "1") return;

        (Tensor cos, Tensor sin) = GetGpuTables(backend);
        backend.WanRopeInterleaved(q, cos, sin, _cachedSeqLen, numHeads, _headDim);
        backend.WanRopeInterleaved(k, cos, sin, _cachedSeqLen, numKvHeads, _headDim);
    }

    /// <summary>Builds (or returns the per-backend cached) duplicated-pair cos/sin tables <c>[S, headDim]</c> from
    /// the host <see cref="Precompute"/> tables — a pure copy-expansion, no trig — and preloads them into
    /// <paramref name="backend"/>'s weight cache so the per-block op skips the H2D upload. All backends' cached
    /// tables are evicted (freed against their OWNING backend, not the caller) the first time this is called
    /// after Precompute refreshes the host tables; each backend then rebuilds lazily on its own next call.
    /// Locked: two backends can call this concurrently (CFG-branch parallelism) or in sequence (a block-range
    /// shard handing off from one backend to another mid-step) — either way each backend gets its own tensors,
    /// never another backend's.</summary>
    private unsafe (Tensor cos, Tensor sin) GetGpuTables(IBackend backend)
    {
        lock (_gpuLock)
        {
            if (_gpuTablesDirty)
            {
                foreach (KeyValuePair<IBackend, (Tensor Cos, Tensor Sin)> entry in _gpuTables)
                {
                    entry.Key.FreeWeights([entry.Value.Cos, entry.Value.Sin]);
                    entry.Value.Cos.Dispose();
                    entry.Value.Sin.Dispose();
                }
                _gpuTables.Clear();
                _gpuTablesDirty = false;
            }
            if (_gpuTables.TryGetValue(backend, out (Tensor Cos, Tensor Sin) cached))
                return (cached.Cos, cached.Sin);

            int halfDim = _headDim / 2;
            Tensor cos = new Tensor(new TensorShape(_cachedSeqLen, _headDim), DType.F32);
            Tensor sin = new Tensor(new TensorShape(_cachedSeqLen, _headDim), DType.F32);
            float* cp = (float*)cos.DataPointer;
            float* sp = (float*)sin.DataPointer;
            for (int s = 0; s < _cachedSeqLen; s++)
            {
                long srcOff = (long)s * halfDim;
                long dstOff = (long)s * _headDim;
                for (int i = 0; i < halfDim; i++)
                {
                    cp[dstOff + 2 * i] = _cosCache![srcOff + i]; cp[dstOff + 2 * i + 1] = _cosCache[srcOff + i];
                    sp[dstOff + 2 * i] = _sinCache![srcOff + i]; sp[dstOff + 2 * i + 1] = _sinCache[srcOff + i];
                }
            }

            backend.PreloadWeights([cos, sin]);
            _gpuTables[backend] = (cos, sin);
            return (cos, sin);
        }
    }

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

        // DIAGNOSTIC (HARTSY_SKIP_ROPE=1): skip rope entirely to isolate the D2H-sync-barrier cost from GPU work.
        if (Environment.GetEnvironmentVariable("HARTSY_SKIP_ROPE") == "1") return;

        int halfDim = _headDim / 2;
        // q/k are [B, numHeads, seqLen, headDim]; each (b,h,s) vector rotates independently → parallelize over
        // the flattened b*h*s space. Single-threaded this loop dominates DiT inference (billions of element-ops
        // per forward across all blocks); identical math, so verified models (Flux/Ideogram) stay bit-exact.
        nint qBase = (nint)q.DataPointer, kBase = (nint)k.DataPointer;
        long rows = (long)batch * numHeads * seqLen;
        fixed (float* cosPtr = _cosCache, sinPtr = _sinCache)
        {
            nint cosBase = (nint)cosPtr, sinBase = (nint)sinPtr;
            System.Threading.Tasks.Parallel.For(0, rows, row =>
            {
                long s = row % seqLen;
                int vecOffset = (int)(row * _headDim);
                int freqIdx = (int)(s * halfDim);
                ApplyRotation((float*)qBase + vecOffset, (float*)cosBase + freqIdx, (float*)sinBase + freqIdx, halfDim);
                ApplyRotation((float*)kBase + vecOffset, (float*)cosBase + freqIdx, (float*)sinBase + freqIdx, halfDim);
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
            System.Threading.Tasks.Parallel.For(0, rows, row =>
            {
                long s = row % seqLen;
                int vecOffset = (int)(row * _headDim);
                ApplyRotation((float*)ptrBase + vecOffset, (float*)cosBase + (int)(s * halfDim), (float*)sinBase + (int)(s * halfDim), halfDim);
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
