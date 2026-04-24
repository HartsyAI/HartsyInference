using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Vae;

/// <summary>ResNet block for VAE encoder/decoder. GroupNorm→SiLU→Conv3x3→GroupNorm→SiLU→Conv3x3 with skip connection. No time embedding (VAE has temb_channels=None).</summary>
public sealed class ResNetBlock2D
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _normGroups;
    private readonly float _normEps;
    private readonly bool _hasShortcut;

    // norm1 parameters
    private Tensor? _norm1Weight;
    private Tensor? _norm1Bias;

    // conv1 parameters (3x3, padding=1)
    private Tensor? _conv1Weight;
    private Tensor? _conv1Bias;

    // norm2 parameters
    private Tensor? _norm2Weight;
    private Tensor? _norm2Bias;

    // conv2 parameters (3x3, padding=1)
    private Tensor? _conv2Weight;
    private Tensor? _conv2Bias;

    // shortcut conv (1x1) when in_channels != out_channels
    private Tensor? _shortcutWeight;
    private Tensor? _shortcutBias;

    /// <summary>Creates a ResNet block with the specified channel configuration.</summary>
    public ResNetBlock2D(int inChannels, int outChannels, int normGroups = 32, float normEps = 1e-6f)
    {
        _inChannels = inChannels;
        _outChannels = outChannels;
        _normGroups = normGroups;
        _normEps = normEps;
        _hasShortcut = inChannels != outChannels;
    }

    /// <summary>Loads weights from a dictionary of named tensors using the given prefix (e.g., "decoder.up_blocks.0.resnets.0").</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _norm1Weight = weights[$"{prefix}.norm1.weight"];
        _norm1Bias = weights[$"{prefix}.norm1.bias"];
        _conv1Weight = weights[$"{prefix}.conv1.weight"];
        _conv1Bias = weights[$"{prefix}.conv1.bias"];
        _norm2Weight = weights[$"{prefix}.norm2.weight"];
        _norm2Bias = weights[$"{prefix}.norm2.bias"];
        _conv2Weight = weights[$"{prefix}.conv2.weight"];
        _conv2Bias = weights[$"{prefix}.conv2.bias"];

        if (_hasShortcut)
        {
            _shortcutWeight = weights[$"{prefix}.conv_shortcut.weight"];
            _shortcutBias = weights[$"{prefix}.conv_shortcut.bias"];
        }
    }

    /// <summary>Enumerates all weight tensors held by this module.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_norm1Weight is not null) yield return _norm1Weight;
        if (_norm1Bias is not null) yield return _norm1Bias;
        if (_conv1Weight is not null) yield return _conv1Weight;
        if (_conv1Bias is not null) yield return _conv1Bias;
        if (_norm2Weight is not null) yield return _norm2Weight;
        if (_norm2Bias is not null) yield return _norm2Bias;
        if (_conv2Weight is not null) yield return _conv2Weight;
        if (_conv2Bias is not null) yield return _conv2Bias;
        if (_shortcutWeight is not null) yield return _shortcutWeight;
        if (_shortcutBias is not null) yield return _shortcutBias;
    }

    /// <summary>Forward pass: input [B, inCh, H, W] → output [B, outCh, H, W].</summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        int batch = (int)input.Shape[0];
        int height = (int)input.Shape[2];
        int width = (int)input.Shape[3];

        // Path 1: norm1 → SiLU → conv1
        TensorShape hiddenShape = new TensorShape(batch, _inChannels, height, width);
        Tensor norm1Out = new Tensor(hiddenShape, DType.F32);
        backend.GroupNorm(norm1Out, input, _norm1Weight!, _norm1Bias!, _normGroups, _normEps);

        Tensor silu1Out = new Tensor(hiddenShape, DType.F32);
        backend.Silu(silu1Out, norm1Out);
        norm1Out.Dispose();

        TensorShape conv1OutShape = new TensorShape(batch, _outChannels, height, width);
        Tensor conv1Out = new Tensor(conv1OutShape, DType.F32);
        backend.Conv2D(conv1Out, silu1Out, _conv1Weight!, _conv1Bias, 1, 1, 1, 1);
        silu1Out.Dispose();

        // Path 1 continued: norm2 → SiLU → conv2
        Tensor norm2Out = new Tensor(conv1OutShape, DType.F32);
        backend.GroupNorm(norm2Out, conv1Out, _norm2Weight!, _norm2Bias!, _normGroups, _normEps);
        conv1Out.Dispose();

        Tensor silu2Out = new Tensor(conv1OutShape, DType.F32);
        backend.Silu(silu2Out, norm2Out);
        norm2Out.Dispose();

        Tensor conv2Out = new Tensor(conv1OutShape, DType.F32);
        backend.Conv2D(conv2Out, silu2Out, _conv2Weight!, _conv2Bias, 1, 1, 1, 1);
        silu2Out.Dispose();

        // Skip connection
        Tensor skip;
        if (_hasShortcut)
        {
            skip = new Tensor(conv1OutShape, DType.F32);
            backend.Conv2D(skip, input, _shortcutWeight!, _shortcutBias, 1, 1, 0, 0);
        }
        else
        {
            skip = input;
        }

        // Residual addition: output = conv2_out + skip
        Tensor output = new Tensor(conv1OutShape, DType.F32);
        backend.Add(output, conv2Out, skip);
        conv2Out.Dispose();

        if (_hasShortcut)
        {
            skip.Dispose();
        }

        return output;
    }
}
