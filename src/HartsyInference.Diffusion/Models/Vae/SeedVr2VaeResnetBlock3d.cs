using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>SeedVR2 VAE resnet: per-frame GroupNorm→SiLU→k3 causal conv, twice, plus (1×1×1 conv_shortcut)
/// residual. Same key layout as HunyuanVideo (<c>{prefix}.{norm1,conv1,norm2,conv2,conv_shortcut}</c>) but
/// per-frame norm stats and zero spatial padding — see <see cref="SeedVr2VaeOps"/>.</summary>
public sealed class SeedVr2VaeResnetBlock3d
{
    private readonly int _groups;
    private readonly float _eps;
    private readonly DType _actDtype;
    private Tensor _norm1W = null!, _norm1B = null!, _norm2W = null!, _norm2B = null!;
    private CausalConv3d _conv1 = null!, _conv2 = null!;
    private CausalConv3d? _shortcut;

    /// <summary>Creates the block; weights arrive via <see cref="LoadWeights"/>.</summary>
    public SeedVr2VaeResnetBlock3d(int groups, float eps, DType? actDtype = null)
    {
        _groups = groups;
        _eps = eps;
        _actDtype = actDtype ?? DType.F32;
    }

    /// <summary>Binds <c>{prefix}.norm1/conv1/norm2/conv2[/conv_shortcut]</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _norm1W = weights[$"{prefix}.norm1.weight"];
        _norm1B = weights[$"{prefix}.norm1.bias"];
        _norm2W = weights[$"{prefix}.norm2.weight"];
        _norm2B = weights[$"{prefix}.norm2.bias"];
        _conv1 = SeedVr2VaeOps.Conv(weights, $"{prefix}.conv1", computeDtype: _actDtype);
        _conv2 = SeedVr2VaeOps.Conv(weights, $"{prefix}.conv2", computeDtype: _actDtype);
        _shortcut = weights.ContainsKey($"{prefix}.conv_shortcut.weight")
            ? SeedVr2VaeOps.Conv(weights, $"{prefix}.conv_shortcut", computeDtype: _actDtype)
            : null;
    }

    /// <summary>Forward on <c>[B,C,T,H,W]</c> F32.</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        Tensor n1 = SeedVr2VaeOps.NormSiluPerFrame(backend, x, _norm1W, _norm1B, _groups, _eps);
        Tensor c1 = _conv1.Forward(backend, n1);
        n1.Dispose();
        Tensor n2 = SeedVr2VaeOps.NormSiluPerFrame(backend, c1, _norm2W, _norm2B, _groups, _eps);
        c1.Dispose();
        Tensor c2 = _conv2.Forward(backend, n2);
        n2.Dispose();
        Tensor residual = _shortcut is null ? x : _shortcut.Forward(backend, x);
        Tensor output = new Tensor(c2.Shape, c2.DType);
        backend.Add(output, c2, residual);
        c2.Dispose();
        if (_shortcut is not null)
            residual.Dispose();
        return output;
    }

    /// <summary>Weights for backend preload/free.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        yield return _norm1W;
        yield return _norm1B;
        yield return _norm2W;
        yield return _norm2B;
        foreach (Tensor t in _conv1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _conv2.EnumerateWeights()) yield return t;
        if (_shortcut is not null)
            foreach (Tensor t in _shortcut.EnumerateWeights()) yield return t;
    }
}
