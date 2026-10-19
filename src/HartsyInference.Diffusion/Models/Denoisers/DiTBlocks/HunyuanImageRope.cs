using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>2-axis rotary position embedding for Hunyuan Image 2.1 (<c>HunyuanImageRotaryPosEmbed</c>). Splits the per-head dimension into <c>(height, width)</c> sub-bands (default <c>[64, 64]</c>) and rotates each pair of features by <c>theta = position * (1 / base^(2k/axisDim))</c>. Applied to image tokens only — text tokens are kept unrotated and concatenated after rotation. Each token at packed-grid <c>(row, col)</c> uses position <c>[row, col]</c>; both axes contribute <c>axesDim/2</c> cos/sin pairs to the head vector. Diffusers uses <c>theta=256</c> by default and <c>get_1d_rotary_pos_embed</c> with <c>use_real=True</c> producing <c>(cos, sin)</c> tables that are concatenated along the feature axis before being applied as the standard <c>(real, imag)</c> 2x2 rotation; this implementation produces the same numerical output by composing the per-axis tables in the same axis order.</summary>
/// <remarks>Frequencies and tables come from <see cref="RopeFrequencyPlan"/> and <see cref="MultiAxisRopeTables"/>. Grid positions
/// come from <see cref="HunyuanImagePosEmbed"/>. Device tables go through a <see cref="RopeTableCache"/> with one lease per
/// backend, so each DiT shard keeps its own tables. The theta is a float, but the legacy arithmetic already widens it to double
/// before <c>Math.Pow</c>, so the standard reciprocal order matches exactly.</remarks>
public sealed unsafe class HunyuanImageRope : IDisposable
{
    private static readonly RopeTableSettings DeviceTableSettings = new()
    {
        AngleOrder = RopeAngleOrder.MultiplyByInverse,
        Layout = RopeTableLayout.InterleavedPair,
        Preload = true,
        RetainedSignatures = 1,
    };

    private readonly int[] _axesDim;
    private readonly int _headDim;
    private readonly RopeFrequencyPlan _plan;
    private readonly RopeTableCache _deviceTables;

    // One held lease per backend: the tables a backend's block loop reads stay valid until it builds a different grid. Another
    // backend's build never frees them, and the lock serializes the held state.
    private readonly object _gate = new();
    private readonly Dictionary<IBackend, HeldGrid> _held = new(ReferenceEqualityComparer.Instance);

    /// <summary>Creates a HunyuanImageRope.</summary>
    /// <param name="axesDim">Per-axis dim split. 2 axes <c>(height, width)</c> for image (default <c>[64, 64]</c>); 3 axes <c>(temporal, height, width)</c> for video (e.g. <c>[16, 56, 56]</c> for HunyuanVideo). Each must be even.</param>
    /// <param name="theta">RoPE base frequency. Default 256.</param>
    public HunyuanImageRope(int[]? axesDim = null, float theta = 256.0f)
    {
        _axesDim = axesDim ?? [64, 64];
        if (_axesDim.Length < 1)
            throw new ArgumentException("HunyuanImageRope requires at least 1 axis.", nameof(axesDim));
        _plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = _axesDim, Theta = theta });
        _headDim = _plan.HeadDim;
        _deviceTables = new RopeTableCache(_plan, DeviceTableSettings);
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
        int seqLen = HunyuanImagePosEmbed.TokenCount(packedDims);
        int pairs = _plan.Pairs;
        float[] cosTable = new float[seqLen * pairs];
        float[] sinTable = new float[seqLen * pairs];
        MultiAxisRopeTables.ComputePairs(_plan, HunyuanImagePosEmbed.GridRows(packedDims), seqLen, RopeAngleOrder.MultiplyByInverse,
            cosTable, sinTable);
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

    /// <summary>Releases the held device tables and the cache. Tables still under a lease are freed by their last release.</summary>
    public void Dispose()
    {
        lock (_gate)
        {
            HeldGrid[] all = _held.Values.ToArray();
            _held.Clear();
            foreach (HeldGrid held in all)
                held.Lease.Dispose();
        }
        _deviceTables.Dispose();
    }

    /// <summary>Returns the duplicated-pair cos/sin tables <c>[S, headDim]</c> for <paramref name="packedDims"/> on
    /// <paramref name="backend"/>, the layout <see cref="IBackend.WanRopeInterleaved"/> reads. Each backend holds one grid;
    /// a grid change releases that backend's previous lease after the new one is acquired.</summary>
    private (Tensor Cos, Tensor Sin, int SeqLen) GetTables(IBackend backend, ReadOnlySpan<int> packedDims)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (packedDims.Length != _axesDim.Length)
            throw new ArgumentException($"packedDims has {packedDims.Length} dims but rope has {_axesDim.Length} axes.", nameof(packedDims));
        int seqLen = HunyuanImagePosEmbed.TokenCount(packedDims);

        lock (_gate)
        {
            if (_held.TryGetValue(backend, out HeldGrid? previous) && packedDims.SequenceEqual(previous.Dims))
                return (previous.Cos, previous.Sin, seqLen);

            RopeTableLease lease = _deviceTables.Acquire(backend, 0, seqLen, HunyuanImagePosEmbed.GridRows(packedDims));
            Tensor cos;
            Tensor sin;
            try
            {
                cos = lease.DeviceCos ?? throw new InvalidOperationException("RoPE device tables require a preloading cache.");
                sin = lease.DeviceSin ?? throw new InvalidOperationException("RoPE device tables require a preloading cache.");
            }
            catch
            {
                lease.Dispose();
                throw;
            }

            _held[backend] = new HeldGrid(packedDims.ToArray(), lease, cos, sin);
            previous?.Lease.Dispose();
            return (cos, sin, seqLen);
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

    /// <summary>The grid a backend currently holds, with its lease and device tensors.</summary>
    private sealed class HeldGrid
    {
        public HeldGrid(int[] dims, RopeTableLease lease, Tensor cos, Tensor sin)
        {
            Dims = dims;
            Lease = lease;
            Cos = cos;
            Sin = sin;
        }

        public int[] Dims { get; }

        public RopeTableLease Lease { get; }

        public Tensor Cos { get; }

        public Tensor Sin { get; }
    }
}
