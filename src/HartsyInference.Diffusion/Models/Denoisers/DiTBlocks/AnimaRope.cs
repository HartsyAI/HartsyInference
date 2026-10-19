using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>3-axis rotary positional embedding for Anima / Cosmos-Predict2 (<c>CosmosRotaryPosEmbed</c>). Splits the head
/// dimension into <c>(t, h, w)</c> sub-bands using Cosmos's allocation scheme:
/// <code>
///     dim_h = (hidden_size // 6) * 2
///     dim_w = (hidden_size // 6) * 2
///     dim_t = hidden_size - dim_h - dim_w
/// </code>
/// Each axis applies a per-axis NTK frequency scaling (<c>theta * scale^(d/(d-2))</c>) before computing inverse frequencies,
/// then concatenates <c>[emb_t, emb_h, emb_w]</c> twice along the last dim so the final cos/sin tables are <c>head_dim</c>-wide
/// with the second half repeating the first (Cosmos's <c>torch.cat([emb_t, emb_h, emb_w] * 2, dim=-1)</c>).
///
/// <para>The rotation is **GPT-NeoX rotate-half** (<c>use_real=True</c>, <c>use_real_unbind_dim=-2</c>): the head dim is split
/// into two halves of <c>head_dim/2</c> elements; the first half is treated as the "real" component, the second as "imag", and
/// the rotation is <c>(x_real * cos - x_imag * sin, x_real * sin + x_imag * cos)</c>. Because the upstream concat duplicates
/// freqs into both halves, both halves see the same cos/sin pattern and the rotation is well-defined.</para>
///
/// <para>Host pairs are computed in float to match the upstream arithmetic. Device rows live in a <see cref="RopeTableCache"/>
/// with the rotate-half layout, and this class supplies the float host writer.</para>
///
/// <para>For the t2i variant (<c>T = 1</c>), the temporal axis evaluates to position 0 and contributes <c>cos=1, sin=0</c> for
/// every <c>dim_t/2</c> frequency — i.e. it leaves Q/K untouched in the temporal sub-band. We compute and store these zeros
/// anyway so the layout matches the upstream tables byte-for-byte (useful for diff dumps).</para></summary>
public sealed unsafe class AnimaRope
{
    private static readonly RopeTableSettings DeviceTableSettings = new()
    {
        AngleOrder = RopeAngleOrder.MultiplyByInverse,
        Layout = RopeTableLayout.RotateHalf,
        Preload = true,
        RetainedSignatures = 1,
    };

    private readonly int _headDim;
    private readonly int _dimT;
    private readonly int _dimH;
    private readonly int _dimW;
    private readonly float _theta;
    private readonly (float T, float H, float W) _ntkFactor;
    private readonly float[] _invFreqT;
    private readonly float[] _invFreqH;
    private readonly float[] _invFreqW;
    private readonly RopeTableCache _deviceTables;
    private readonly object _gpuTableLock = new();
    private readonly Dictionary<IBackend, HeldTables> _held = new(ReferenceEqualityComparer.Instance);
    private (int T, int H, int W) _gridSignature;
    private float[] _gridPositions = [];

    private sealed record HeldTables(int TFrames, int HPatched, int WPatched, RopeTableLease Lease);

    /// <summary>Creates a 3-axis Cosmos RoPE for the given <paramref name="hiddenSize"/> per head and base period.
    /// <paramref name="ropeScale"/> applies the upstream NTK scale: <c>theta_axis = theta * scale^(dim_axis/(dim_axis-2))</c>
    /// for axes with <c>dim_axis > 2</c>. <paramref name="beforeRetire"/> runs for a backend before its device tables are freed.</summary>
    public AnimaRope(int hiddenSize, float theta, (float T, float H, float W) ropeScale, Action<IBackend>? beforeRetire = null)
    {
        if (hiddenSize <= 0 || (hiddenSize & 1) != 0)
            throw new ArgumentOutOfRangeException(
                nameof(hiddenSize), hiddenSize, "AnimaRope head dimension must be positive and even.");
        if (!float.IsFinite(theta) || theta <= 0f)
            throw new ArgumentOutOfRangeException(nameof(theta), theta, "AnimaRope theta must be finite and positive.");
        if (!float.IsFinite(ropeScale.T) || ropeScale.T <= 0f || !float.IsFinite(ropeScale.H) || ropeScale.H <= 0f
            || !float.IsFinite(ropeScale.W) || ropeScale.W <= 0f)
            throw new ArgumentOutOfRangeException(
                nameof(ropeScale), ropeScale, "AnimaRope per-axis scales must be finite and positive.");
        _headDim = hiddenSize;
        _dimH = (hiddenSize / 6) * 2;
        _dimW = (hiddenSize / 6) * 2;
        _dimT = hiddenSize - _dimH - _dimW;
        if (_dimT % 2 != 0 || _dimH % 2 != 0 || _dimW % 2 != 0)
            throw new ArgumentException(
                $"AnimaRope: each axis dim must be even. Got dim_t={_dimT}, dim_h={_dimH}, dim_w={_dimW} for hidden={hiddenSize}.",
                nameof(hiddenSize));
        _theta = theta;
        _ntkFactor = (
            T: NtkFactor(ropeScale.T, _dimT),
            H: NtkFactor(ropeScale.H, _dimH),
            W: NtkFactor(ropeScale.W, _dimW));
        _invFreqT = InverseFrequencies(_dimT, _theta * _ntkFactor.T);
        _invFreqH = InverseFrequencies(_dimH, _theta * _ntkFactor.H);
        _invFreqW = InverseFrequencies(_dimW, _theta * _ntkFactor.W);
        RopeFrequencyPlan plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [_dimT, _dimH, _dimW], Theta = theta });
        _deviceTables = new RopeTableCache(plan, DeviceTableSettings, beforeRetire, WriteHostPairs);
    }

    /// <summary>Total head dim (= sum of axis sub-dims).</summary>
    public int HeadDim => _headDim;

    /// <summary>Per-axis sub-dim split <c>(dim_t, dim_h, dim_w)</c>.</summary>
    public (int T, int H, int W) AxisDim => (_dimT, _dimH, _dimW);

    /// <summary>Computes one cos/sin table per token for an image grid of shape <c>(tFrames, hPatched, wPatched)</c>. For
    /// t2i, callers pass <c>tFrames=1</c>. Returns <see cref="Tensor"/> arrays sized <c>[seqLen, headDim]</c> each,
    /// row-major over <c>(t, h, w)</c>. Caller takes ownership and disposes.</summary>
    public (Tensor Cos, Tensor Sin) BuildFreqs(int tFrames, int hPatched, int wPatched)
    {
        ValidateGrid(tFrames, hPatched, wPatched);
        int seqLen;
        try
        {
            seqLen = checked(checked(tFrames * hPatched) * wPatched);
            _ = checked((long)seqLen * _headDim * sizeof(float));
        }
        catch (OverflowException)
        {
            throw new ArgumentOutOfRangeException(
                nameof(wPatched), "AnimaRope table geometry exceeds signed addressable storage.");
        }

        float[] positions = GridPositions(tFrames, hPatched, wPatched);
        int halfDim = _headDim / 2;
        float[] cos = new float[checked(seqLen * halfDim)];
        float[] sin = new float[cos.Length];
        WriteHostPairs(positions, seqLen, cos, sin);

        TensorShape shape = new(seqLen, _headDim);
        Tensor? cosTensor = null;
        Tensor? sinTensor = null;
        bool succeeded = false;
        try
        {
            cosTensor = new Tensor(shape, DType.F32);
            sinTensor = new Tensor(shape, DType.F32);
            int count = checked(seqLen * _headDim);
            RopeTableExpander.Expand(cos, sin, seqLen, _headDim, RopeTableLayout.RotateHalf, 0, RopeTablePadding.Identity,
                new Span<float>(cosTensor.DataPointer, count), new Span<float>(sinTensor.DataPointer, count));
            succeeded = true;
            return (cosTensor, sinTensor);
        }
        finally
        {
            if (!succeeded)
            {
                cosTensor?.Dispose();
                sinTensor?.Dispose();
            }
        }
    }

    /// <summary>Returns this backend's device table pair for the grid. A lease held for the same grid is reused, so a denoising
    /// step never re-acquires. Borrowed: callers must not dispose the tensors, and must pair the model phase with
    /// <see cref="ReleaseGpuTables"/>.</summary>
    internal (Tensor Cos, Tensor Sin) GetOrCreateTables(IBackend backend, int tFrames, int hPatched, int wPatched)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ValidateGrid(tFrames, hPatched, wPatched);
        lock (_gpuTableLock)
        {
            if (_held.TryGetValue(backend, out HeldTables? held))
            {
                if (held.TFrames == tFrames && held.HPatched == hPatched && held.WPatched == wPatched)
                    return DeviceTablesOf(held.Lease);
                held.Lease.Dispose();
                _held.Remove(backend);
            }

            if (_gridSignature != (tFrames, hPatched, wPatched))
            {
                _gridPositions = GridPositions(tFrames, hPatched, wPatched);
                _gridSignature = (tFrames, hPatched, wPatched);
            }
            RopeTableLease lease = _deviceTables.Acquire(backend, 0, _gridPositions.Length / 3, _gridPositions);
            _held.Add(backend, new HeldTables(tFrames, hPatched, wPatched, lease));
            return DeviceTablesOf(lease);
        }
    }

    /// <summary>Releases this backend's device tables and any lease held for it. Idempotent. A failed free leaves the
    /// tables tracked, so the next call retries.</summary>
    public void ReleaseGpuTables(IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        lock (_gpuTableLock)
        {
            if (_held.Remove(backend, out HeldTables? held))
                held.Lease.Dispose();
            _deviceTables.ReleaseBackend(backend);
        }
    }

    /// <summary>Applies in-place rotate-half rotation to Q and K. Both must be <c>[B, numHeads, seqLen, headDim]</c>
    /// laid out contiguously. <paramref name="cosTable"/> and <paramref name="sinTable"/> must be <c>[seqLen, headDim]</c>
    /// from <see cref="BuildFreqs"/>.</summary>
    public void ApplyRotation(Tensor q, Tensor k, Tensor cosTable, Tensor sinTable, int batch, int numHeads, int seqLen)
    {
        int halfDim = _headDim / 2;
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;
        float* cosPtr = (float*)cosTable.DataPointer;
        float* sinPtr = (float*)sinTable.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    long vecOff = (((long)b * numHeads + h) * seqLen + s) * _headDim;
                    long freqOff = (long)s * _headDim;
                    RotateHalf(qPtr + vecOff, cosPtr + freqOff, sinPtr + freqOff, halfDim);
                    RotateHalf(kPtr + vecOff, cosPtr + freqOff, sinPtr + freqOff, halfDim);
                }
            }
        }
    }

    /// <summary>Cosmos NTK factor: <c>scale^(dim/(dim-2))</c>. Returns 1.0 when <paramref name="dim"/> ≤ 2 to avoid div-by-zero.</summary>
    private static float NtkFactor(float scale, int dim)
    {
        if (dim <= 2 || MathF.Abs(scale - 1.0f) < 1e-12f) return 1.0f;
        return MathF.Pow(scale, (float)dim / (dim - 2));
    }

    /// <summary>Inverse frequencies <c>1 / theta^(2k / dim)</c> for one axis, length <c>dim/2</c>, in float.</summary>
    private static float[] InverseFrequencies(int dim, float theta)
    {
        float[] invFreqs = new float[dim / 2];
        int n = invFreqs.Length;
        for (int k = 0; k < n; k++)
        {
            float exponent = (float)(2 * k) / dim;
            invFreqs[k] = 1.0f / MathF.Pow(theta, exponent);
        }
        return invFreqs;
    }

    /// <summary>Writes host pairs in the rotate-half compact layout: each axis's <c>dim/2</c> values, in <c>(t, h, w)</c> order.</summary>
    private void WriteHostPairs(ReadOnlySpan<float> positions, int tokens, Span<float> cos, Span<float> sin)
    {
        int halfDim = _headDim / 2;
        for (int s = 0; s < tokens; s++)
        {
            int row = s * halfDim;
            int featOff = 0;
            WriteAxisFreqs(cos, sin, row, featOff, positions[s * 3], _invFreqT);
            featOff += _dimT / 2;
            WriteAxisFreqs(cos, sin, row, featOff, positions[s * 3 + 1], _invFreqH);
            featOff += _dimH / 2;
            WriteAxisFreqs(cos, sin, row, featOff, positions[s * 3 + 2], _invFreqW);
        }
    }

    /// <summary>Writes <c>cos(pos * invFreq[k])</c> / <c>sin(pos * invFreq[k])</c> for one axis.</summary>
    private static void WriteAxisFreqs(Span<float> cos, Span<float> sin, int rowOff, int featOff, float pos, ReadOnlySpan<float> invFreqs)
    {
        for (int k = 0; k < invFreqs.Length; k++)
        {
            float angle = pos * invFreqs[k];
            cos[rowOff + featOff + k] = MathF.Cos(angle);
            sin[rowOff + featOff + k] = MathF.Sin(angle);
        }
    }

    /// <summary>Per-token <c>(t, h, w)</c> coordinates for a grid, in the sequence order <c>(t * H + h) * W + w</c>.</summary>
    private static float[] GridPositions(int tFrames, int hPatched, int wPatched)
    {
        int seqLen = checked(tFrames * hPatched * wPatched);
        float[] positions = new float[checked(seqLen * 3)];
        int s = 0;
        for (int t = 0; t < tFrames; t++)
        {
            for (int h = 0; h < hPatched; h++)
            {
                for (int w = 0; w < wPatched; w++)
                {
                    positions[s * 3] = t;
                    positions[s * 3 + 1] = h;
                    positions[s * 3 + 2] = w;
                    s++;
                }
            }
        }
        return positions;
    }

    private static void ValidateGrid(int tFrames, int hPatched, int wPatched)
    {
        if (tFrames <= 0)
            throw new ArgumentOutOfRangeException(nameof(tFrames), tFrames, "AnimaRope frame count must be positive.");
        if (hPatched <= 0)
            throw new ArgumentOutOfRangeException(nameof(hPatched), hPatched, "AnimaRope packed height must be positive.");
        if (wPatched <= 0)
            throw new ArgumentOutOfRangeException(nameof(wPatched), wPatched, "AnimaRope packed width must be positive.");
    }

    private static (Tensor Cos, Tensor Sin) DeviceTablesOf(RopeTableLease lease) =>
        lease.DeviceCos is { } cos && lease.DeviceSin is { } sin
            ? (cos, sin)
            : throw new InvalidOperationException("Anima RoPE device tables require a preloading cache.");

    /// <summary>GPT-NeoX rotate-half: split the head into two halves [real | imag], rotate by (cos, sin) per element pairwise.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RotateHalf(float* vec, float* cos, float* sin, int halfDim)
    {
        for (int d = 0; d < halfDim; d++)
        {
            float xReal = vec[d];
            float xImag = vec[halfDim + d];
            vec[d] = xReal * cos[d] - xImag * sin[d];
            vec[halfDim + d] = xReal * sin[halfDim + d] + xImag * cos[halfDim + d];
        }
    }
}
