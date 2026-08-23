using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Oobleck;

/// <summary>Oobleck residual unit (diffusers <c>OobleckResidualUnit</c>) — SnakeBeta → dilated WNConv1d(k7) → SnakeBeta → WNConv1d(k1) → residual add.</summary>
/// <remarks>
/// <code>
///   y = SnakeBeta(x)                                  # logscale alpha/beta snake
///   y = WNConv1d(dim, dim, k=7, dilation=d, padding=(7-1)*d/2)(y)
///   y = SnakeBeta(y)
///   y = WNConv1d(dim, dim, k=1)(y)
///   return x + y                                      # symmetric pad keeps T equal
/// </code>
///
/// <para>Differs from <c>DacResidualUnit</c> in the activation: Oobleck's Snake1d stores LOG-scale
/// <c>alpha</c>/<c>beta</c> parameters of shape <c>[1, dim, 1]</c> — both are exponentiated once at
/// load time, then fed to <see cref="IBackend.Snake"/>'s snake-beta path (divisor = beta + ε).</para>
///
/// <para>State-dict keys (real ACE-Step 1.5 / descript <c>nn.Sequential</c> layout — the residual
/// unit is itself a <c>Sequential</c> of [Snake, WNConv1d k=7, Snake, WNConv1d k=1]):
/// <c>{prefix}.layers.0.{alpha,beta}</c>, <c>{prefix}.layers.1.{weight_g,weight_v,bias}</c>,
/// <c>{prefix}.layers.2.{alpha,beta}</c>, <c>{prefix}.layers.3.{weight_g,weight_v,bias}</c>.</para></remarks>
internal sealed class OobleckResidualUnit
{
    private const int Kernel = 7;

    private readonly string _prefix;
    private readonly int _dim;
    private readonly int _dilation;

    private Tensor? _snake1Alpha, _snake1Beta;
    private Tensor? _snake2Alpha, _snake2Beta;
    private Tensor? _conv1W, _conv1B;
    private Tensor? _conv2W, _conv2B;

    public OobleckResidualUnit(string prefix, int dim, int dilation)
    {
        _prefix = prefix;
        _dim = dim;
        _dilation = dilation;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        (_snake1Alpha, _snake1Beta) = OobleckOps.LoadSnake(w, $"{_prefix}.layers.0", _dim);
        _conv1W = WeightNormFusion.LoadFused(w, $"{_prefix}.layers.1");
        _conv1B = WhisperOps.EnsureF32(w[$"{_prefix}.layers.1.bias"]);
        (_snake2Alpha, _snake2Beta) = OobleckOps.LoadSnake(w, $"{_prefix}.layers.2", _dim);
        _conv2W = WeightNormFusion.LoadFused(w, $"{_prefix}.layers.3");
        _conv2B = WhisperOps.EnsureF32(w[$"{_prefix}.layers.3.bias"]);
    }

    /// <summary>Forward — channels-first <c>[B, dim, T]</c>. Returns a fresh tensor; input is NOT disposed.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int batch, int t)
    {
        if (_conv1W is null) throw new InvalidOperationException($"OobleckResidualUnit '{_prefix}' weights not loaded.");

        Tensor a1 = new(x.Shape, DType.F32);
        backend.Snake(a1, x, _snake1Alpha!, _snake1Beta);

        // Symmetric padding (k-1)*d/2 keeps T unchanged (k=7 → (k-1)*d even for any d).
        int pad = (Kernel - 1) * _dilation / 2;
        Tensor mid = new(new TensorShape(batch, _dim, t), DType.F32);
        backend.Conv1d(mid, a1, _conv1W!, _conv1B,
            stride: 1, padLeft: pad, padRight: pad, dilation: _dilation, groups: 1);
        a1.Dispose();

        Tensor a2 = new(mid.Shape, DType.F32);
        backend.Snake(a2, mid, _snake2Alpha!, _snake2Beta);
        mid.Dispose();

        Tensor proj = new(new TensorShape(batch, _dim, t), DType.F32);
        backend.Conv1d(proj, a2, _conv2W!, _conv2B,
            stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        a2.Dispose();

        Tensor result = new(proj.Shape, DType.F32);
        backend.Add(result, x, proj);
        proj.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_snake1Alpha, _snake1Beta, _snake2Alpha, _snake2Beta, _conv1W, _conv1B, _conv2W, _conv2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
