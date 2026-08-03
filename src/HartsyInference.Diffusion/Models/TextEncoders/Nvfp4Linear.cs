using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
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
/// <para>Memory: the 50-layer Qwen3-VL-32B tower is ~100 GB at F32, so nothing is materialized at load. Each forward
/// dequantizes one <c>[out, in]</c> F32 slice (≤524 MB), runs the GEMM, then drains the stream and drops any device
/// copy before freeing it — the same discipline <see cref="GptOssMoeFfn"/> established after concurrent reuse of a
/// transient slice corrupted the CUDA cache bookkeeping.</para></summary>
public sealed class Nvfp4Linear : IDisposable
{
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
            ? EnsureF32(pqs)
            : null;

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
        if (globalScale.ElementCount != 1)
            throw new InvalidOperationException($"nvfp4 '{prefix}.weight_scale_2' must be a scalar; got {globalScale.Shape}.");
        ValidatePreQuantScale(preQuantScale, prefix, inFeatures);

        return new Nvfp4Linear(null,
            weight.Reshape(new TensorShape(1, outFeatures, weight.Shape[1])),
            blockScale.Reshape(new TensorShape(1, blockScale.Shape[0], blockScale.Shape[1])),
            globalScale.Reshape(new TensorShape(1)),
            preQuantScale, outFeatures, inFeatures);
    }

    /// <summary>Computes <c>output = input · Wᵀ</c> for <c>input [1, seq, InFeatures]</c> and
    /// <c>output [1, seq, OutFeatures]</c>, applying <c>pre_quant_scale</c> and any transient dequant.</summary>
    public void Forward(IBackend backend, Tensor output, Tensor input)
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

            Tensor slice = new Tensor(new TensorShape(_outFeatures, _inFeatures), DType.F32);
            try
            {
                Nvfp4Codec.DequantExpertSlice(_packed!, _blockScale!, _globalScale!, 0, slice);
                backend.Linear(output, x, slice, null);
            }
            finally
            {
                // Drain, then drop any cached device copy keyed by this tensor before its host memory goes away.
                backend.Sync();
                backend.FreeWeights([slice]);
                slice.Dispose();
            }
        }
        finally
        {
            scaled?.Dispose();
        }
    }

    /// <summary>Enumerates the resident weight tensors for GPU preloading; the transient dequant slices are not
    /// included because they only exist inside <see cref="Forward"/>.</summary>
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

    private static Tensor EnsureF32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    private static void ValidatePreQuantScale(Tensor? preQuantScale, string prefix, int inFeatures)
    {
        if (preQuantScale is null) return;
        if (preQuantScale.ElementCount != inFeatures)
            throw new InvalidOperationException(
                $"'{prefix}.pre_quant_scale' must cover the {inFeatures} input channels; got {preQuantScale.Shape}.");
    }
}
