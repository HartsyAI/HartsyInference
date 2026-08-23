using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Dac;

/// <summary>DAC residual unit — two-conv stack with snake activations interleaved.</summary>
/// <remarks>
/// <code>
///   r = x
///   y = Snake1d(x)                                  # per-channel alpha-snake
///   y = Conv1d(dim, dim, k=7, dilation=d, padding=(7-1)*d/2)(y)
///   y = Snake1d(y)
///   y = Conv1d(dim, dim, k=1)(y)
///   # Center-crop x to match y's reduced length (symmetric padding rounds down):
///   pad = (x.shape[-1] - y.shape[-1]) // 2
///   if pad > 0: x = x[..., pad:-pad]
///   return x + y
/// </code>
///
/// <para>Padding is symmetric (<c>(k-1)*d/2</c> on each side), NOT causal — DAC is
/// offline-only by design. The center-crop handles minor length mismatches when the
/// effective receptive field can't be perfectly halved.</para>
///
/// <para>State-dict keys (PyTorch <c>nn.Sequential</c> ordering inside the unit's
/// <c>self.block</c>):
/// <list type="bullet">
///   <item><c>{prefix}.block.0.alpha</c> — first Snake1d, shape <c>[1, dim, 1]</c></item>
///   <item><c>{prefix}.block.1.weight_g</c> / <c>.weight_v</c> / <c>.bias</c> — first WNConv1d</item>
///   <item><c>{prefix}.block.2.alpha</c> — second Snake1d</item>
///   <item><c>{prefix}.block.3.weight_g</c> / <c>.weight_v</c> / <c>.bias</c> — second WNConv1d (kernel=1)</item>
/// </list></para>
/// </remarks>
internal sealed unsafe class DacResidualUnit
{
    private readonly string _prefix;
    private readonly int _dim;
    private readonly int _kernel;
    private readonly int _dilation;

    private Tensor? _snake1Alpha;
    private Tensor? _snake2Alpha;
    private Tensor? _conv1W;
    private Tensor? _conv1B;
    private Tensor? _conv2W;
    private Tensor? _conv2B;

    public DacResidualUnit(string prefix, int dim, int kernel, int dilation)
    {
        _prefix = prefix;
        _dim = dim;
        _kernel = kernel;
        _dilation = dilation;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        // Snake alpha stored as [1, dim, 1] — reshape to [dim] so backend.Snake's
        // per-channel layout matches.
        _snake1Alpha = WhisperOps.EnsureF32(w[$"{_prefix}.block.0.alpha"]).Reshape(new TensorShape(_dim));
        _conv1W = LoadFusedWeight(w, $"{_prefix}.block.1");
        _conv1B = WhisperOps.EnsureF32(w[$"{_prefix}.block.1.bias"]);
        _snake2Alpha = WhisperOps.EnsureF32(w[$"{_prefix}.block.2.alpha"]).Reshape(new TensorShape(_dim));
        _conv2W = LoadFusedWeight(w, $"{_prefix}.block.3");
        _conv2B = WhisperOps.EnsureF32(w[$"{_prefix}.block.3.bias"]);
    }

    /// <summary>Forward — channels-first <c>[B, dim, T]</c>. Returns a fresh tensor. Input is NOT disposed.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int batch, int t)
    {
        if (_conv1W is null) throw new InvalidOperationException($"DacResidualUnit '{_prefix}' weights not loaded.");

        return SnakeResidualBlock.Forward(backend, x, batch, t, _dim, _kernel, _dilation, groups: 1,
            _snake1Alpha!, _conv1W!, _conv1B, _snake2Alpha!, _conv2W!, _conv2B);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_snake1Alpha, _snake2Alpha, _conv1W, _conv1B, _conv2W, _conv2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    /// <summary>Loads a weight-normed conv weight. DAC uses bare <c>nn.Conv1d</c> wrapped with <c>weight_norm</c>, so the keys live directly on the layer (no nested <c>.conv.conv</c> like EnCodec's SConv1d wrapper).</summary>
    private static Tensor LoadFusedWeight(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        return WeightNormFusion.Compose(w, prefix);
    }
}
