using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Wan2.2 VAE residual block (<c>ResidualBlock</c> in <c>vae2_2.py</c>): <c>RMS_norm → SiLU → CausalConv3d(in→out,3) → RMS_norm → SiLU → CausalConv3d(out→out,3)</c> plus a <c>CausalConv3d(in→out,1)</c> shortcut (identity when in==out). Reuses <see cref="WanRmsNorm"/> + <see cref="CausalConv3d"/>. Stateless offline form (feat_cache=null → zero causal temporal pad); the streaming-cache variant is threaded by the decoder.</summary>
public sealed unsafe class Wan22ResidualBlock
{
    private readonly int _inDim;
    private readonly int _outDim;
    private readonly WanRmsNorm _norm1;
    private readonly WanRmsNorm _norm2;
    private CausalConv3d? _conv1;
    private CausalConv3d? _conv2;
    private CausalConv3d? _shortcut;   // null when in==out (identity)

    public Wan22ResidualBlock(int inDim, int outDim)
    {
        _inDim = inDim;
        _outDim = outDim;
        _norm1 = new WanRmsNorm(inDim);
        _norm2 = new WanRmsNorm(outDim);
    }

    public int OutChannels => _outDim;

    /// <summary>Loads weights. <paramref name="prefix"/> e.g. <c>"decoder.middle.0"</c> — the <c>residual</c> Sequential indices are 0=RMS_norm, 2=conv1, 3=RMS_norm, 6=conv2; <c>shortcut</c> exists only when in≠out.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _norm1.LoadWeights(weights[$"{prefix}.residual.0.gamma"]);
        _conv1 = new CausalConv3d(weights[$"{prefix}.residual.2.weight"], Bias(weights, $"{prefix}.residual.2.bias"),
            padT: 1, padH: 1, padW: 1);
        _norm2.LoadWeights(weights[$"{prefix}.residual.3.gamma"]);
        _conv2 = new CausalConv3d(weights[$"{prefix}.residual.6.weight"], Bias(weights, $"{prefix}.residual.6.bias"),
            padT: 1, padH: 1, padW: 1);
        if (_inDim != _outDim)
            _shortcut = new CausalConv3d(weights[$"{prefix}.shortcut.weight"], Bias(weights, $"{prefix}.shortcut.bias"),
                padT: 0, padH: 0, padW: 0);
    }

    /// <summary>Enumerates weights for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _norm1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _norm2.EnumerateWeights()) yield return t;
        if (_conv1 is not null) foreach (Tensor t in _conv1.EnumerateWeights()) yield return t;
        if (_conv2 is not null) foreach (Tensor t in _conv2.EnumerateWeights()) yield return t;
        if (_shortcut is not null) foreach (Tensor t in _shortcut.EnumerateWeights()) yield return t;
    }

    /// <summary>Forward over <c>[B, inDim, T, H, W]</c> → <c>[B, outDim, T, H, W]</c>. Pass <paramref name="cache"/> for streaming video decode (the two CausalConv3d layers thread their temporal cache through it, in order); null = stateless image/first-chunk path. The 1×1×1 shortcut conv is never cached.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Wan22StreamCache? cache = null)
    {
        // Identity shortcut (in==out): add the untouched input x directly — no host clone. The norm/conv path below
        // allocates fresh tensors and never mutates x, so it is still valid at the residual add.
        Tensor? shortcut = _shortcut?.Forward(backend, x);

        Tensor r1 = _norm1.Forward(backend, x);
        Silu(backend, r1);
        Tensor? cc1 = cache?.StepConv(backend, r1);
        Tensor c1 = _conv1!.Forward(backend, r1, cc1);
        cc1?.Dispose();
        r1.Dispose();
        Tensor r2 = _norm2.Forward(backend, c1);
        c1.Dispose();
        Silu(backend, r2);
        Tensor? cc2 = cache?.StepConv(backend, r2);
        Tensor c2 = _conv2!.Forward(backend, r2, cc2);
        cc2?.Dispose();
        r2.Dispose();

        Tensor outT = new Tensor(c2.Shape, DType.F32);
        backend.Add(outT, c2, shortcut ?? x);
        c2.Dispose();
        shortcut?.Dispose();
        return outT;
    }

    private static void Silu(IBackend backend, Tensor t) => backend.Silu(t, t);

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? b) ? b : null;
}
