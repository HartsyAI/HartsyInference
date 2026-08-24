using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>Activation applied after an <see cref="RtDetrConv"/>.</summary>
public enum RtDetrActivation
{
    /// <summary>Identity (used by the ResNet basic-block second conv, the RepVGG branch convs, and the input projections).</summary>
    None,
    /// <summary>ReLU — the ResNet-vd backbone activation.</summary>
    Relu,
    /// <summary>SiLU — the CCFM (hybrid-encoder) conv activation.</summary>
    Silu,
}

/// <summary>A single Conv2D layer with folded-in BatchNorm (so it carries a bias) and an optional
/// activation. RT-DETR checkpoints store BN as separate <c>running_mean/var/weight/bias</c> buffers;
/// the checkpoint converter folds each BN into its preceding conv, leaving a plain
/// <c>{prefix}.weight</c> / <c>{prefix}.bias</c> pair that this class loads.</summary>
public sealed class RtDetrConv(int strideH, int strideW, int padH, int padW, RtDetrActivation activation)
{
    private readonly int _strideH = strideH, _strideW = strideW;
    private readonly int _padH = padH, _padW = padW;
    private readonly RtDetrActivation _activation = activation;

    private Tensor? _weight;   // [C_out, C_in, kH, kW]
    private Tensor? _bias;     // [C_out]

    /// <summary>Output channel count (available after weights are loaded).</summary>
    public int OutChannels => (int)_weight!.Shape[0];

    /// <summary>Loads the folded conv <c>{prefix}.weight</c> and <c>{prefix}.bias</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _weight = EnsureF32(weights[$"{prefix}.weight"]);
        _bias = EnsureF32(weights[$"{prefix}.bias"]);
    }

    /// <summary>Yields the conv weight + bias for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_weight is not null) yield return _weight;
        if (_bias is not null) yield return _bias;
    }

    /// <summary>Output spatial size for a given input under this conv's stride/padding.</summary>
    public (int H, int W) OutputSpatial(int inH, int inW)
    {
        int kH = (int)_weight!.Shape[2];
        int kW = (int)_weight!.Shape[3];
        int oH = (inH + 2 * _padH - kH) / _strideH + 1;
        int oW = (inW + 2 * _padW - kW) / _strideW + 1;
        return (oH, oW);
    }

    /// <summary>Runs Conv → bias → optional activation. Owns the returned tensor.</summary>
    public unsafe Tensor Forward(IBackend backend, Tensor input)
    {
        if (_weight is null || _bias is null)
            throw new InvalidOperationException("RtDetrConv weights not loaded.");
        int n = (int)input.Shape[0];
        (int oH, int oW) = OutputSpatial((int)input.Shape[2], (int)input.Shape[3]);
        Tensor output = new Tensor(new TensorShape(n, OutChannels, oH, oW), DType.F32);
        backend.Conv2D(output, input, _weight, _bias, _strideH, _strideW, _padH, _padW);
        switch (_activation)
        {
            case RtDetrActivation.Silu:
                backend.Silu(output, output);
                break;
            case RtDetrActivation.Relu:
                Span<float> span = output.AsSpan<float>();
                for (int i = 0; i < span.Length; i++)
                    if (span[i] < 0f) span[i] = 0f;
                break;
        }
        return output;
    }

    private static Tensor EnsureF32(Tensor t) =>
        t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}
