using HartsyInference.Audio.Models.Codecs.Dac;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Snac;

/// <summary>SNAC residual unit. Structurally identical to <see cref="DacResidualUnit"/>: <c>Snake → Conv1d(k=7, dilation) → Snake → Conv1d(k=1) → residual add</c>. Kept as a SNAC-namespaced class so the codecs stay independently maintainable; the math is identical and weights load with the same key conventions.</summary>
internal sealed unsafe class SnacResidualUnit(string prefix, int dim, int kernel, int dilation, int groups = 1)
{
    private readonly string _prefix = prefix;
    private readonly int _dim = dim;
    private readonly int _kernel = kernel;
    private readonly int _dilation = dilation;
    private readonly int _groups = groups;

    private Tensor? _snake1Alpha;
    private Tensor? _snake2Alpha;
    private Tensor? _conv1W;
    private Tensor? _conv1B;
    private Tensor? _conv2W;
    private Tensor? _conv2B;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _snake1Alpha = WhisperOps.EnsureF32(w[$"{_prefix}.block.0.alpha"]).Reshape(new TensorShape(_dim));
        _conv1W = WeightNormFusion.LoadFused(w, $"{_prefix}.block.1");
        _conv1B = WhisperOps.EnsureF32(w[$"{_prefix}.block.1.bias"]);
        _snake2Alpha = WhisperOps.EnsureF32(w[$"{_prefix}.block.2.alpha"]).Reshape(new TensorShape(_dim));
        _conv2W = WeightNormFusion.LoadFused(w, $"{_prefix}.block.3");
        _conv2B = WhisperOps.EnsureF32(w[$"{_prefix}.block.3.bias"]);
    }

    public Tensor Forward(IBackend backend, Tensor x, int batch, int t)
    {
        if (_conv1W is null) throw new InvalidOperationException($"SnacResidualUnit '{_prefix}' weights not loaded.");

        return SnakeResidualBlock.Forward(backend, x, batch, t, _dim, _kernel, _dilation, _groups,
            _snake1Alpha!, _conv1W!, _conv1B, _snake2Alpha!, _conv2W!, _conv2B);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_snake1Alpha, _snake2Alpha, _conv1W, _conv1B, _conv2W, _conv2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
