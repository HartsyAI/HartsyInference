using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Codecs.Dac;

/// <summary>DAC residual unit. Two-conv stack with snake activations interleaved:
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
/// </list></para></summary>
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

    /// <summary>Forward — channels-first <c>[B, dim, T]</c>. Returns a fresh tensor.
    /// Input is NOT disposed.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int batch, int t)
    {
        if (_conv1W is null) throw new InvalidOperationException($"DacResidualUnit '{_prefix}' weights not loaded.");

        // Path: Snake → Conv1d(k=7, dilation) → Snake → Conv1d(k=1).
        Tensor a1 = new(x.Shape, DType.F32);
        backend.Snake(a1, x, _snake1Alpha!, null);

        // Symmetric padding for kernel=7 dilation=d: pad each side by (k-1)*d/2.
        int pad = (_kernel - 1) * _dilation / 2;
        int tConv1 = t + 2 * pad - _dilation * (_kernel - 1);
        Tensor mid = new(new TensorShape(batch, _dim, tConv1), DType.F32);
        backend.Conv1d(mid, a1, _conv1W!, _conv1B,
            stride: 1, padLeft: pad, padRight: pad, dilation: _dilation, groups: 1);
        a1.Dispose();

        Tensor a2 = new(mid.Shape, DType.F32);
        backend.Snake(a2, mid, _snake2Alpha!, null);
        mid.Dispose();

        // Kernel=1, no padding — preserves time.
        Tensor proj = new(new TensorShape(batch, _dim, tConv1), DType.F32);
        backend.Conv1d(proj, a2, _conv2W!, _conv2B,
            stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        a2.Dispose();

        // Center-crop x to proj's time dim if necessary (DAC trims the residual to
        // match the reduced length when the dilated conv eats a few samples).
        int tProj = (int)proj.Shape[2];
        int diff = t - tProj;
        int cropLeft = diff / 2;
        Tensor result = new(proj.Shape, DType.F32);

        if (diff > 0)
        {
            float* xp = (float*)x.DataPointer;
            float* pp = (float*)proj.DataPointer;
            float* rp = (float*)result.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < _dim; c++)
                {
                    int srcXBase = (b * _dim + c) * t + cropLeft;
                    int srcPBase = (b * _dim + c) * tProj;
                    int dstBase = (b * _dim + c) * tProj;
                    for (int j = 0; j < tProj; j++) rp[dstBase + j] = xp[srcXBase + j] + pp[srcPBase + j];
                }
        }
        else
        {
            backend.Add(result, x, proj);
        }
        proj.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_snake1Alpha, _snake2Alpha, _conv1W, _conv1B, _conv2W, _conv2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    /// <summary>Loads a weight-normed conv weight. DAC uses bare <c>nn.Conv1d</c> wrapped
    /// with <c>weight_norm</c>, so the keys live directly on the layer (no nested
    /// <c>.conv.conv</c> like EnCodec's SConv1d wrapper).</summary>
    private static Tensor LoadFusedWeight(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        Tensor g = WhisperOps.EnsureF32(w[$"{prefix}.weight_g"]);
        Tensor v = WhisperOps.EnsureF32(w[$"{prefix}.weight_v"]);
        return WeightNormFusion.Fuse(g, v);
    }
}
