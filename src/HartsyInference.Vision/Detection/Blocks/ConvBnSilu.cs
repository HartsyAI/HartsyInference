using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.Blocks;

/// <summary>YOLO's basic building block: Conv2D → BatchNorm → SiLU. At conversion time the BN
/// affine + running statistics are folded into the Conv weights and bias, so at runtime this is
/// just <c>Conv2D(w, b)</c> followed by <c>SiLU</c>. See the Python conversion script for the
/// folding math:
/// <code>
/// w_folded[c_out, ...] = w_conv[c_out, ...] * (gamma[c_out] / sqrt(var[c_out] + eps))
/// b_folded[c_out]      = beta[c_out] - mean[c_out] * (gamma[c_out] / sqrt(var[c_out] + eps))
/// </code>
/// <para>Optionally <see cref="UseSilu"/> can be set <c>false</c> to skip the activation — the
/// detection head's final 1×1 projection convs (<c>cv2[2]</c> / <c>cv3[2]</c>) have no
/// BN-and-SiLU; only the two preceding 3×3 convs do.</para></summary>
public sealed class ConvBnSilu
{
    private readonly int _outChannels;
    private readonly int _strideH, _strideW;
    private readonly int _padH, _padW;
    private readonly bool _useSilu;

    private Tensor? _weight;     // [C_out, C_in, kH, kW]
    private Tensor? _bias;       // [C_out]

    /// <summary>Output channel count.</summary>
    public int OutChannels => _outChannels;

    /// <summary>Whether to apply SiLU after the conv (false for the final 1×1 projection convs in the detect head).</summary>
    public bool UseSilu => _useSilu;

    /// <summary>Creates a Conv-BN-SiLU block. Padding is "same" only when strideH/W=1 and padH/W = (kernel-1)/2 — the caller is responsible for picking consistent stride+padding combinations.</summary>
    public ConvBnSilu(int outChannels, int strideH = 1, int strideW = 1, int padH = 0, int padW = 0, bool useSilu = true)
    {
        if (outChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(outChannels), outChannels, "Out channels must be positive.");
        _outChannels = outChannels;
        _strideH = strideH;
        _strideW = strideW;
        _padH = padH;
        _padW = padW;
        _useSilu = useSilu;
    }

    /// <summary>Loads conv weights and bias from a checkpoint. Keys are <c>{prefix}.weight</c> and <c>{prefix}.bias</c> — after BN folding the Ultralytics naming becomes <c>module.conv.weight</c>/<c>module.conv.bias</c> so callers typically pass <c>"...conv"</c> as the prefix.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _weight = EnsureF32(weights[$"{prefix}.weight"]);
        _bias = EnsureF32(weights[$"{prefix}.bias"]);
    }

    /// <summary>Yields the conv weight + bias for GPU preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_weight is not null) yield return _weight;
        if (_bias is not null) yield return _bias;
    }

    /// <summary>Output shape for a given input H/W under this block's stride and padding.</summary>
    public (int outH, int outW) OutputSpatial(int inH, int inW)
    {
        int kH = (int)_weight!.Shape[2];
        int kW = (int)_weight!.Shape[3];
        int oH = (inH + 2 * _padH - kH) / _strideH + 1;
        int oW = (inW + 2 * _padW - kW) / _strideW + 1;
        return (oH, oW);
    }

    /// <summary>Runs Conv → bias → optional SiLU. Owns the returned tensor.</summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        if (_weight is null || _bias is null)
            throw new InvalidOperationException("ConvBnSilu weights not loaded.");

        int n = (int)input.Shape[0];
        (int oH, int oW) = OutputSpatial((int)input.Shape[2], (int)input.Shape[3]);
        Tensor output = new Tensor(new TensorShape(n, _outChannels, oH, oW), DType.F32);
        backend.Conv2D(output, input, _weight, _bias, _strideH, _strideW, _padH, _padW);
        if (_useSilu)
            backend.Silu(output, output);
        return output;
    }

    private static Tensor EnsureF32(Tensor t) =>
        t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}
