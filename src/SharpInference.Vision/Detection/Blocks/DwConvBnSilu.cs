using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Vision.Detection.Blocks;

/// <summary>Depthwise variant of <see cref="ConvBnSilu"/> — groups equal channel count, so each
/// output channel sees exactly one input channel. After BN folding the runtime path is
/// <c>Conv2dDepthwise(w, b) → optional SiLU</c>. Used by YOLO11's classification branch (two
/// depthwise-3×3 stages followed by 1×1 pointwise projections) and by
/// <see cref="PsaAttention"/>'s positional-encoding conv.</summary>
public sealed class DwConvBnSilu
{
    private readonly int _channels;
    private readonly int _strideH, _strideW;
    private readonly int _padH, _padW;
    private readonly bool _useSilu;

    private Tensor? _weight; // [C, 1, kH, kW]
    private Tensor? _bias;   // [C]

    public int Channels => _channels;

    public DwConvBnSilu(int channels, int strideH = 1, int strideW = 1, int padH = 0, int padW = 0, bool useSilu = true)
    {
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels), channels, "Channels must be positive.");
        _channels = channels;
        _strideH = strideH;
        _strideW = strideW;
        _padH = padH;
        _padW = padW;
        _useSilu = useSilu;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _weight = EnsureF32(weights[$"{prefix}.weight"]);
        _bias = EnsureF32(weights[$"{prefix}.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_weight is not null) yield return _weight;
        if (_bias is not null) yield return _bias;
    }

    public Tensor Forward(IBackend backend, Tensor input)
    {
        if (_weight is null || _bias is null)
            throw new InvalidOperationException("DwConvBnSilu weights not loaded.");

        int n = (int)input.Shape[0];
        int iH = (int)input.Shape[2];
        int iW = (int)input.Shape[3];
        int kH = (int)_weight.Shape[2];
        int kW = (int)_weight.Shape[3];
        int oH = (iH + 2 * _padH - kH) / _strideH + 1;
        int oW = (iW + 2 * _padW - kW) / _strideW + 1;

        Tensor output = new Tensor(new TensorShape(n, _channels, oH, oW), DType.F32);
        backend.Conv2dDepthwise(output, input, _weight, _bias, _strideH, _strideW, _padH, _padW);
        if (_useSilu)
            backend.Silu(output, output);
        return output;
    }

    private static Tensor EnsureF32(Tensor t) =>
        t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}
