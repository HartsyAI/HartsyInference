using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Rotary position embedding for Kandinsky 5. Diffusers stores the rope as 2x2 rotation
/// matrices laid out as <c>[cos, -sin, sin, cos]</c> and applies them via <c>(rope * x.unsqueeze(-1)).sum(-1)</c>
/// — mathematically equivalent to the standard <c>[cos, -sin; sin, cos]</c> 2x2 rotation on consecutive
/// pairs of values along the head dimension.
///
/// Two flavors:
/// <list type="bullet">
/// <item><b>1D</b> (text stream): every text token has a single integer position; pair frequencies
/// span the full <c>head_dim</c> with <c>freqs[k] = 1 / theta^(2k/head_dim)</c> for <c>k ∈ [0, head_dim/2)</c>.</item>
/// <item><b>3D</b> (visual stream): the head dim is split into three contiguous slices sized by
/// <c>axes_dims = [d_t, d_h, d_w]</c>. Each axis has its own <c>get_freqs(d_axis/2)</c> table indexed
/// by the per-axis position. Concatenated rope = [t-rope, h-rope, w-rope].</item>
/// </list>
/// In both cases the resulting cos/sin tables are <c>[seq_len, head_dim/2]</c>; <see cref="ApplyRotation"/>
/// rotates each consecutive pair of values <c>(vec[2i], vec[2i+1])</c> by the corresponding angle.
///
/// Host pairs are computed by <see cref="Kandinsky5PosEmbed"/>, which owns the coordinate scales. The device rows are
/// duplicated pairs in a <see cref="RopeTableCache"/> whose host writer copies those pairs.</summary>
public sealed unsafe class Kandinsky5Rope : IQkRotation, IHeadMajorQkRotation
{
    private static readonly RopeTableSettings DeviceTableSettings = new()
    {
        AngleOrder = RopeAngleOrder.MultiplyByInverse,
        Layout = RopeTableLayout.InterleavedPair,
        Preload = true,
        RetainedSignatures = 1,
    };

    private readonly int _headDim;
    private readonly float _theta;

    private float[]? _cosCache;
    private float[]? _sinCache;
    private int _cachedSeqLen;
    private string? _precomputeKey;

    // Device rows. The cache is bound to one axis layout, so it is recreated when the layout changes. Every new precompute
    // clears it, so the rows always come from the latest host pairs.
    private RopeTableCache? _deviceTables;
    private int[]? _deviceAxes;
    private float[]? _signature;

    /// <summary>Creates a Kandinsky 5 rope with the given head dim and base period.</summary>
    public Kandinsky5Rope(int headDim, float theta = 10000.0f)
    {
        if (headDim % 2 != 0)
            throw new ArgumentException($"Kandinsky5Rope head_dim must be even, got {headDim}", nameof(headDim));
        _headDim = headDim;
        _theta = theta;
    }

    /// <summary>Total head dimension covered by this rope.</summary>
    public int HeadDim => _headDim;

    /// <summary>Precomputes the 1D RoPE tables for a sequence of integer positions. Mirrors <c>args = pos[:, None] * freqs[None, :]</c> with <c>freqs = get_freqs(head_dim/2, theta)</c>.</summary>
    /// <param name="positions">Per-token positions, shape <c>[seqLen]</c> as int32 (we read as int32 ptr).</param>
    public void Precompute1D(ReadOnlySpan<int> positions)
    {
        // The transformer precomputes every forward with identical inputs (same seq every step) —
        // memoize so the host trig AND the GPU table upload happen once per generation, not per step.
        string key = $"1d:{positions.Length}:{(positions.Length > 0 ? positions[0] : 0)}:{(positions.Length > 0 ? positions[^1] : 0)}";
        if (key == _precomputeKey && _cosCache is not null) return;
        _precomputeKey = key;

        int seqLen = positions.Length;
        int halfDim = _headDim / 2;
        float[] cos = new float[seqLen * halfDim];
        float[] sin = new float[cos.Length];
        Kandinsky5PosEmbed.Fill1D(_theta, _headDim, positions, cos, sin);
        Publish([_headDim], cos, sin, seqLen, Kandinsky5PosEmbed.Signature1D(positions));
    }

    /// <summary>Precomputes the 3D RoPE tables for visual tokens flattened in <c>(t, h, w)</c> order. Each axis contributes <c>axesDims[i]/2</c> frequency pairs concatenated along the head dim. Position grid is <c>[duration, height, width]</c> in patch units (after <c>patch_size</c>).</summary>
    /// <param name="axesDims">Per-axis dim split (e.g. <c>[32, 48, 48]</c>). Must sum to <see cref="HeadDim"/>.</param>
    /// <param name="duration">Number of temporal patches (1 for image t2i).</param>
    /// <param name="height">Number of vertical patches.</param>
    /// <param name="width">Number of horizontal patches.</param>
    /// <param name="tStart">Starting position offset along the temporal axis (default 0).</param>
    /// <param name="scaleT">Temporal rope arg divisor (video resolution-dependent; 1.0 = T2I behavior).</param>
    /// <param name="scaleH">Vertical rope arg divisor — diffusers <c>args_h / scale_factor[1]</c>.</param>
    /// <param name="scaleW">Horizontal rope arg divisor — diffusers <c>args_w / scale_factor[2]</c>.</param>
    public void Precompute3D(int[] axesDims, int duration, int height, int width, int tStart = 0,
        float scaleT = 1.0f, float scaleH = 1.0f, float scaleW = 1.0f)
    {
        if (scaleT <= 0f || scaleH <= 0f || scaleW <= 0f)
            throw new ArgumentException($"RoPE scale factors must be positive, got ({scaleT}, {scaleH}, {scaleW}).");
        int axisSum = 0;
        for (int i = 0; i < axesDims.Length; i++) axisSum += axesDims[i];
        if (axisSum != _headDim)
            throw new ArgumentException(
                $"axesDims sum ({axisSum}) must equal head_dim ({_headDim})", nameof(axesDims));

        // Memoized like Precompute1D — the per-step inputs are identical across the denoise loop.
        string key = $"3d:{string.Join(',', axesDims)}:{duration}:{height}:{width}:{tStart}:{scaleT}:{scaleH}:{scaleW}";
        if (key == _precomputeKey && _cosCache is not null) return;
        _precomputeKey = key;

        int seqLen = duration * height * width;
        int halfDim = _headDim / 2;
        float[] cos = new float[seqLen * halfDim];
        float[] sin = new float[cos.Length];
        Kandinsky5PosEmbed.Fill3D(_theta, axesDims, _headDim, duration, height, width, tStart, scaleT, scaleH, scaleW, cos, sin);
        Publish(axesDims, cos, sin, seqLen, Kandinsky5PosEmbed.Signature3D(duration, height, width, tStart));
    }

    /// <summary>Applies RoPE rotation to Q and K in-place. Q/K are <c>[B, numHeads, seqLen, headDim]</c> contiguous floats. Each pair <c>(x[2i], x[2i+1])</c> is rotated by <c>(cos[i], sin[i])</c>.</summary>
    public void Apply(Tensor q, Tensor k, int batch, int numHeads, int seqLen)
    {
        if (_cosCache is null || _sinCache is null)
            throw new InvalidOperationException("Kandinsky5Rope: Precompute must be called before Apply.");
        if (seqLen != _cachedSeqLen)
            throw new InvalidOperationException(
                $"Kandinsky5Rope: cached seq_len {_cachedSeqLen} doesn't match requested {seqLen}");

        int halfDim = _headDim / 2;
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;

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
                        ApplyRotation(qPtr + vecOffset, cosPtr + freqIdx, sinPtr + freqIdx, halfDim);
                        ApplyRotation(kPtr + vecOffset, cosPtr + freqIdx, sinPtr + freqIdx, halfDim);
                    }
                }
            }
        }
    }

    /// <summary>GPU-resident RoPE on a PRE-head-permute <c>[1, S, H, D]</c> Q/K pair (byte-identical to the <c>[S, heads·headDim]</c> layout <see cref="IBackend.WanRopeInterleaved"/> expects) — the rotation math is the same interleaved GPT-J pairing, so the Wan kernel applies verbatim. Replaces the per-block host excursion (a full Q/K D2H → trig → H2D round trip per attention, the dominant per-step cost). The duplicated-pair cos/sin tables are built ONCE per Precompute and preloaded to the GPU weight cache. Requires batch 1; callers keep <see cref="Apply"/> as the batched fallback.</summary>
    public void ApplyGpu(IBackend backend, Tensor q, Tensor k, int numHeads, int seqLen)
    {
        if (_cosCache is null || _sinCache is null || _deviceTables is null || _signature is null)
            throw new InvalidOperationException("Kandinsky5Rope: Precompute must be called before ApplyGpu.");
        if (seqLen != _cachedSeqLen)
            throw new InvalidOperationException(
                $"Kandinsky5Rope: cached seq_len {_cachedSeqLen} doesn't match requested {seqLen}");

        using RopeTableLease lease = _deviceTables.Acquire(backend, 0, seqLen, _signature);
        (Tensor cos, Tensor sin) = DeviceTablesOf(lease);
        backend.WanRopeInterleaved(q, cos, sin, seqLen, numHeads, _headDim);
        backend.WanRopeInterleaved(k, cos, sin, seqLen, numHeads, _headDim);
    }

    /// <summary>Installs new host pairs and retires the device rows built from the previous ones.</summary>
    private void Publish(int[] axes, float[] cos, float[] sin, int seqLen, float[] signature)
    {
        if (_deviceTables is null || _deviceAxes is null || !_deviceAxes.AsSpan().SequenceEqual(axes))
        {
            RopeFrequencyPlan plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = axes, Theta = _theta });
            RopeTableCache next = new(plan, DeviceTableSettings, hostWriter: (_, tokens, c, s) => CopyHostPairs(tokens, c, s));
            _deviceTables?.Dispose();
            _deviceTables = next;
            _deviceAxes = (int[])axes.Clone();
        }
        else
        {
            _deviceTables.Clear();
        }

        _cosCache = cos;
        _sinCache = sin;
        _cachedSeqLen = seqLen;
        _signature = signature;
    }

    /// <summary>Host writer for the device cache: copies the pairs this rope computed for the current precompute.</summary>
    private void CopyHostPairs(int tokens, Span<float> cos, Span<float> sin)
    {
        if (_cosCache is null || _sinCache is null || tokens != _cachedSeqLen)
            throw new InvalidOperationException("Kandinsky5Rope: host pairs changed while device rows were being built.");
        _cosCache.AsSpan().CopyTo(cos);
        _sinCache.AsSpan().CopyTo(sin);
    }

    private static (Tensor Cos, Tensor Sin) DeviceTablesOf(RopeTableLease lease) =>
        lease.DeviceCos is { } cos && lease.DeviceSin is { } sin
            ? (cos, sin)
            : throw new InvalidOperationException("Kandinsky5Rope device tables require a preloading cache.");

    /// <summary>Rotates each consecutive pair via <c>x[2i]'   = cos·x[2i]   - sin·x[2i+1]</c> and <c>x[2i+1]' = sin·x[2i] + cos·x[2i+1]</c>. Equivalent to the diffusers' <c>(rope * x.unsqueeze(-1)).sum(-1)</c> with rope viewed as <c>[c, -s; s, c]</c>.</summary>
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

    /// <inheritdoc/>
    void IQkRotation.RotateTokenMajor(IBackend backend, Tensor q, Tensor k, int heads, int kvHeads) =>
        ApplyGpu(backend, q, k, heads, (int)q.Shape[1]);

    /// <inheritdoc/>
    void IHeadMajorQkRotation.RotateHeadMajor(IBackend backend, Tensor q, Tensor k, int batch, int heads, int kvHeads, int tokens) =>
        Apply(q, k, batch, heads, tokens);
}
