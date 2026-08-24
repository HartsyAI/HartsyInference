using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.Nvfp4;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>One bias-free linear layer whose weight is either dense or stored in ComfyUI's <c>nvfp4</c> AWQ packing,
/// dequantized transiently per forward so the packed bank stays mmap-backed.
///
/// <para><b>Packed form</b> (verified against <c>qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors</c>):
/// <c>{name}.weight</c> U8 <c>[out, in/2]</c> nibble-packed E2M1, <c>{name}.weight_scale</c> F8-E4M3
/// <c>[out, in/16]</c> in NVIDIA's <see cref="HartsyInference.ModelAssets.BlockScale.BlockScaleSwizzle">blocked</see>
/// layout, <c>{name}.weight_scale_2</c> F32 scalar. The rank-2 tensors are reshaped to the rank-3
/// <c>[1, …]</c> bank shape <see cref="Nvfp4Codec.DequantExpertSlice"/> consumes, so one codec serves both.</para>
///
/// <para><b>AWQ.</b> The quantizer divided each input channel by a scale before quantizing. For most linears that
/// scale was migrated into the preceding RMSNorm weight (the checkpoint's own norms are already adjusted, so nothing
/// is needed at runtime); <c>o_proj</c> and <c>down_proj</c> have no such host and instead ship
/// <c>{name}.pre_quant_scale</c> over the INPUT dim, which must multiply the activation:
/// <c>x·Wᵀ = (x⊙s)·(W/s)ᵀ</c>. Confirmed numerically against the BF16 release of the same checkpoint —
/// <c>amax(W/s per 16-element block) / (6·scale)</c> lands in E4M3's ±6% rounding band, while <c>W·s</c> and the
/// unswizzled scale reading do not.</para>
///
/// <para><b>Memory and transfer.</b> The 50-layer Qwen3-VL-32B tower is ~100 GB at F32 and ~49 GB at 16-bit, so
/// nothing is materialized at load and no dequantized weight can be cached device-side (the tower must also share a
/// 24 GB card with a ~19.4 GB DiT). Each forward dequantizes one <c>[out, in]</c> slice, runs the GEMM, then drops
/// any device copy before releasing it — the discipline <see cref="GptOssMoeFfn"/> established after concurrent
/// reuse of a transient slice corrupted the CUDA cache bookkeeping. Two things make that transient cheap:
/// <list type="bullet">
/// <item>The slice is <b>BF16</b>, not F32 — half the host-to-device bytes (~49 GB instead of ~97 GB per encode) at
/// a storage error of 2⁻⁹ against E2M1's own ~25% worst-case half-step. BF16 rather than F16 because
/// <c>ResolveGemmDtype</c> then runs the GEMM in BF16, and F16 overflows on the SwiGLU <c>gated</c> tensor that
/// <c>down_proj</c> consumes.</item>
/// <item>The host buffer comes from a caller-owned scratch (see <see cref="CreateDequantScratch"/>) shared by every
/// layer, so a 50-layer encode allocates one buffer instead of 350. The <see cref="Tensor"/> wrapper is still fresh
/// per call: reusing a Tensor <i>object</i> whose bytes change is what corrupts the reference-keyed device caches.</item>
/// </list></para></summary>
public sealed unsafe class Nvfp4Linear : IDisposable
{
    /// <summary>Dtype the packed weight is dequantized to for the GEMM.</summary>
    public static readonly DType DequantDType = DType.BF16;

    /// <summary>E4M3 byte → float, derived from the shared <see cref="Tensor.CastTo"/> converter because the codec's
    /// own table is internal to ModelAssets. 256 entries covers every byte, so the decode is a lookup.</summary>
    private static readonly float[] E4M3Decode = BuildE4M3Decode();

    private readonly Tensor? _dense;
    private readonly Tensor? _packed;
    private readonly Tensor? _blockScale;
    private readonly Tensor? _globalScale;
    private readonly Tensor? _preQuantScale;
    private readonly int _outFeatures;
    private readonly int _inFeatures;
    private int _disposed;

    private Nvfp4Linear(Tensor? dense, Tensor? packed, Tensor? blockScale, Tensor? globalScale,
        Tensor? preQuantScale, int outFeatures, int inFeatures)
    {
        _dense = dense;
        _packed = packed;
        _blockScale = blockScale;
        _globalScale = globalScale;
        _preQuantScale = preQuantScale;
        _outFeatures = outFeatures;
        _inFeatures = inFeatures;
    }

    /// <summary>Output channel count.</summary>
    public int OutFeatures => _outFeatures;

    /// <summary>Input channel count.</summary>
    public int InFeatures => _inFeatures;

    /// <summary>True when the weight is nvfp4-packed and dequantized per forward.</summary>
    public bool IsPacked => _packed is not null;

    /// <summary>Elements a shared dequant scratch must hold to serve this layer; zero when the weight is dense.</summary>
    public long DequantScratchElements => _packed is null ? 0 : (long)_outFeatures * _inFeatures;

    /// <summary>Allocates the scratch <see cref="Forward"/> borrows its dequant slice from, sized by the largest
    /// <see cref="DequantScratchElements"/> across the layers that will share it.</summary>
    public static Tensor CreateDequantScratch(long elementCount)
    {
        if (elementCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(elementCount), "Dequant scratch must hold at least one element.");
        return new Tensor(new TensorShape(elementCount), DequantDType);
    }

    /// <summary>Binds <c>{prefix}.weight</c> and, when it is U8, its nvfp4 scale companions plus any
    /// <c>pre_quant_scale</c>. Throws when the weight is missing or a companion of a packed weight is absent.</summary>
    public static Nvfp4Linear Load(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (!weights.TryGetValue($"{prefix}.weight", out Tensor? weight))
            throw new InvalidOperationException($"Missing linear weight '{prefix}.weight'.");
        if (weight.Shape.Rank != 2)
            throw new InvalidOperationException($"Linear '{prefix}.weight' must be rank-2; got {weight.Shape}.");

        Tensor? preQuantScale = weights.TryGetValue($"{prefix}.pre_quant_scale", out Tensor? pqs)
            ? TensorCasts.EnsureF32(pqs) : null;

        if (weight.DType != DType.U8)
        {
            int denseIn = (int)weight.Shape[1];
            ValidatePreQuantScale(preQuantScale, prefix, denseIn);
            return new Nvfp4Linear(weight, null, null, null, preQuantScale, (int)weight.Shape[0], denseIn);
        }

        if (!weights.TryGetValue($"{prefix}.weight_scale", out Tensor? blockScale) ||
            !weights.TryGetValue($"{prefix}.weight_scale_2", out Tensor? globalScale))
        {
            throw new InvalidOperationException(
                $"nvfp4 weight '{prefix}.weight' is missing '{prefix}.weight_scale' or '{prefix}.weight_scale_2'.");
        }

        int outFeatures = (int)weight.Shape[0];
        int inFeatures = (int)weight.Shape[1] * 2;
        // Rows may be padded up to a multiple of 128 by the blocked layout; columns never are for these shapes.
        if (blockScale.Shape.Rank != 2 || blockScale.Shape[0] < outFeatures ||
            blockScale.Shape[1] != inFeatures / Nvfp4Codec.GroupSize)
        {
            throw new InvalidOperationException(
                $"nvfp4 '{prefix}.weight_scale' must be [>={outFeatures}, {inFeatures / Nvfp4Codec.GroupSize}]; " +
                $"got {blockScale.Shape}.");
        }
        if (blockScale.DType != DType.F8E4M3)
            throw new InvalidOperationException($"nvfp4 '{prefix}.weight_scale' must be F8E4M3; got {blockScale.DType}.");
        if (globalScale.DType != DType.F32 || globalScale.ElementCount != 1)
            throw new InvalidOperationException($"nvfp4 '{prefix}.weight_scale_2' must be an F32 scalar; got {globalScale.DType} {globalScale.Shape}.");
        ValidatePreQuantScale(preQuantScale, prefix, inFeatures);

        return new Nvfp4Linear(null,
            weight.Reshape(new TensorShape(1, outFeatures, weight.Shape[1])),
            blockScale.Reshape(new TensorShape(1, blockScale.Shape[0], blockScale.Shape[1])),
            globalScale.Reshape(new TensorShape(1)),
            preQuantScale, outFeatures, inFeatures);
    }

    /// <summary>Computes <c>output = input · Wᵀ</c> for <c>input [1, seq, InFeatures]</c> and
    /// <c>output [1, seq, OutFeatures]</c>, applying <c>pre_quant_scale</c> and any transient dequant.</summary>
    /// <param name="dequantScratch">Optional shared buffer from <see cref="CreateDequantScratch"/> to dequantize
    /// into; null allocates (and frees) one per call.</param>
    public void Forward(IBackend backend, Tensor output, Tensor input, Tensor? dequantScratch = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (input.Shape.Rank != 3 || input.Shape[0] != 1 || input.Shape[2] != _inFeatures)
            throw new ArgumentException($"Expected input [1, seq, {_inFeatures}]; got {input.Shape}.", nameof(input));
        if (output.Shape.Rank != 3 || output.Shape[0] != 1 || output.Shape[1] != input.Shape[1] ||
            output.Shape[2] != _outFeatures)
            throw new ArgumentException($"Expected output [1, {input.Shape[1]}, {_outFeatures}]; got {output.Shape}.", nameof(output));

        Tensor? scaled = null;
        try
        {
            Tensor x = input;
            if (_preQuantScale is not null)
            {
                scaled = new Tensor(input.Shape, DType.F32);
                backend.AffineBroadcastLastDim(scaled, input, _preQuantScale, null);
                x = scaled;
            }

            if (_dense is not null)
            {
                backend.Linear(output, x, _dense, null);
                return;
            }

            Tensor slice = BorrowSlice(dequantScratch);
            try
            {
                DequantBf16(_packed!, _blockScale!, _globalScale!, slice);
                backend.Linear(output, x, slice, null);
            }
            finally
            {
                // Drops any cached device copy keyed by this tensor, and drains the stream on the way (both the CUDA
                // and Vulkan helpers wait first), so the scratch bytes are free to be overwritten by the next layer.
                backend.FreeWeights([slice]);
                slice.Dispose();
            }
        }
        finally
        {
            scaled?.Dispose();
        }
    }

    /// <summary>Enumerates the resident weight tensors for GPU preloading; the packed bank and its scales are
    /// excluded because the dequant reads them on the HOST — preloading would put a device-cached tensor in front of
    /// those reads and evict it on every layer.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_dense is not null) yield return _dense;
        if (_preQuantScale is not null) yield return _preQuantScale;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _packed?.Dispose();
        _blockScale?.Dispose();
        _globalScale?.Dispose();
    }

    /// <summary>Wraps the scratch in a FRESH tensor object each call — the device caches are reference-keyed, so a
    /// reused object whose bytes changed would serve the previous layer's weight.</summary>
    private Tensor BorrowSlice(Tensor? dequantScratch)
    {
        TensorShape shape = new TensorShape(_outFeatures, _inFeatures);
        if (dequantScratch is null)
            return new Tensor(shape, DequantDType);
        if (dequantScratch.DType != DequantDType || dequantScratch.ElementCount < shape.ElementCount)
            throw new ArgumentException(
                $"Dequant scratch must be {DequantDType} with at least {shape.ElementCount} elements; " +
                $"got {dequantScratch.DType} with {dequantScratch.ElementCount}.", nameof(dequantScratch));
        return new Tensor(dequantScratch.DataPointer, shape, DequantDType);
    }

    /// <summary>Dequantizes the single-expert bank into <paramref name="destination"/> as BF16 <c>[out, in]</c> — the
    /// <c>[N, K]</c> layout <c>IBackend.Linear</c> consumes. Same arithmetic as
    /// <see cref="Nvfp4Codec.DequantExpertSlice"/>, which only writes F32; folding the narrowing into the dequant
    /// avoids a second full-size host pass over the weight (its natural home is an overload on the codec).</summary>
    private static void DequantBf16(Tensor packed, Tensor blockScale, Tensor globalScale, Tensor destination)
    {
        long outDim = packed.Shape[1];
        long inHalf = packed.Shape[2];
        if (destination.DType != DequantDType || destination.Shape.Rank != 2 ||
            destination.Shape[0] != outDim || destination.Shape[1] != inHalf * 2)
            throw new ArgumentException(
                $"Destination must be {DequantDType} [{outDim}, {inHalf * 2}]; got {destination.DType} {destination.Shape}.",
                nameof(destination));

        DequantBf16Core((byte*)packed.DataPointer, (byte*)blockScale.DataPointer,
            blockScale.Fp8ScaleFactor, ((float*)globalScale.DataPointer)[0],
            outDim, inHalf, blockScale.Shape[2], (ushort*)destination.DataPointer);
    }

    /// <summary>Row-major BF16 dequant, parallel over on-disk rows: packed reads and BF16 writes are both
    /// sequential and the per-16-element block scale is decoded once per group.</summary>
    private static void DequantBf16Core(byte* w, byte* bs, float scaleFactor, float global,
        long outDim, long inHalf, long paddedCols, ushort* dst)
    {
        float[] lut = Nvfp4Codec.E2M1Lut;
        float[] e4m3 = E4M3Decode;
        int bytesPerGroup = Nvfp4Codec.GroupSize / 2;
        long inDim = inHalf * 2;

        Parallel.For(0, (int)outDim, r =>
        {
            byte* wr = w + (long)r * inHalf;
            ushort* dr = dst + (long)r * inDim;
            float scale = 0f;
            for (long k = 0; k < inHalf; k++)
            {
                if (k % bytesPerGroup == 0)
                    scale = e4m3[bs[BlockScaleSwizzle.SwizzledIndex(r, k / bytesPerGroup, paddedCols)]] * scaleFactor * global;
                byte packed = wr[k];
                dr[2 * k] = TensorCasts.F32ToBf16Bits(lut[(packed >> 4) & 0x0F] * scale);   // high nibble = even element
                dr[2 * k + 1] = TensorCasts.F32ToBf16Bits(lut[packed & 0x0F] * scale);      // low nibble = odd element
            }
        });
    }

    private static float[] BuildE4M3Decode()
    {
        using Tensor bytes = new Tensor(new TensorShape(256), DType.F8E4M3);
        byte* p = (byte*)bytes.DataPointer;
        for (int i = 0; i < 256; i++) p[i] = (byte)i;
        using Tensor decoded = bytes.CastTo(DType.F32);
        return decoded.AsReadOnlySpan<float>().ToArray();
    }

    private static void ValidatePreQuantScale(Tensor? preQuantScale, string prefix, int inFeatures)
    {
        if (preQuantScale is null) return;
        if (preQuantScale.ElementCount != inFeatures)
            throw new InvalidOperationException(
                $"'{prefix}.pre_quant_scale' must cover the {inFeatures} input channels; got {preQuantScale.Shape}.");
    }
}
