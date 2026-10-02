using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Auk;

/// <summary>AuK double-stream (MMDiT) block: separate adaLN, qkv and FFN per stream with one joint attention over [audio, text].</summary>
public sealed class AukDoubleBlock(AukConfig config)
{
    private readonly AukAttnProj _attnX = new();
    private readonly AukAttnProj _attnC = new();
    private readonly AukSwiGluFfn _ffX = new();
    private readonly AukSwiGluFfn _ffC = new();
    private Tensor? _modXW, _modXB, _modCW, _modCB;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        int dim = config.Dim;
        _modXW = AukOps.Take(weights, $"{prefix}.attn_norm_x.linear.weight", 6 * dim, dim);
        _modXB = AukOps.Take(weights, $"{prefix}.attn_norm_x.linear.bias", 6 * dim);
        _modCW = AukOps.Take(weights, $"{prefix}.attn_norm_c.linear.weight", 6 * dim, dim);
        _modCB = AukOps.Take(weights, $"{prefix}.attn_norm_c.linear.bias", 6 * dim);
        _attnX.Load(weights, $"{prefix}.attn", "to_qkv", "q_norm", "k_norm", "to_out.0", dim, config.Heads);
        _attnC.Load(weights, $"{prefix}.attn", "to_qkv_c", "c_q_norm", "c_k_norm", "to_out_c", dim, config.Heads);
        _ffX.Load(weights, $"{prefix}.ff_x", dim, config.FfInner);
        _ffC.Load(weights, $"{prefix}.ff_c", dim, config.FfInner);
    }

    /// <summary>Returns new (text, audio) streams; inputs stay with the caller. Both streams rotate at positions starting at 0 of the shared tables.</summary>
    public (Tensor C, Tensor X) Forward(IBackend backend, Tensor x, Tensor c, Tensor siluTime, int n, int nt, Tensor cos, Tensor sin)
    {
        int dim = config.Dim;
        int heads = config.Heads;
        int headDim = config.HeadDim;
        using AukModulation modC = new(backend, siluTime, _modCW!, _modCB!, dim);
        using AukModulation modX = new(backend, siluTime, _modXW!, _modXB!, dim);

        Tensor normX = AukOps.AdaNorm(backend, x, modX.ScaleMsa, modX.ShiftMsa, config.NormEps);
        Tensor normC = AukOps.AdaNorm(backend, c, modC.ScaleMsa, modC.ShiftMsa, config.NormEps);
        (Tensor qx, Tensor kx, Tensor vx) = _attnX.Project(backend, normX, n, cos, sin);
        (Tensor qc, Tensor kc, Tensor vc) = _attnC.Project(backend, normC, nt, cos, sin);
        normX.Dispose(); normC.Dispose();

        int total = n + nt;
        TensorShape joint = new(1, total, heads, headDim);
        Tensor q = Join(backend, joint, qx, qc);
        Tensor k = Join(backend, joint, kx, kc);
        Tensor v = Join(backend, joint, vx, vc);
        qx.Dispose(); qc.Dispose(); kx.Dispose(); kc.Dispose(); vx.Dispose(); vc.Dispose();
        Tensor merged = AukOps.Attend(backend, q, k, v, total, heads, headDim);
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor mergedX = new(new TensorShape(1, n, dim), DType.F32);
        Tensor mergedC = new(new TensorShape(1, nt, dim), DType.F32);
        backend.Split([mergedX, mergedC], merged, 1);
        merged.Dispose();
        Tensor attnX = _attnX.Output(backend, mergedX, n);
        Tensor attnC = _attnC.Output(backend, mergedC, nt);
        mergedX.Dispose(); mergedC.Dispose();

        Tensor cMid = AukOps.GatedResidual(backend, c, attnC, modC.GateMsa);
        Tensor xMid = AukOps.GatedResidual(backend, x, attnX, modX.GateMsa);
        attnC.Dispose(); attnX.Dispose();
        Tensor cOut = _ffC.ForwardResidual(backend, cMid, modC, nt, config.NormEps);
        Tensor xOut = _ffX.ForwardResidual(backend, xMid, modX, n, config.NormEps);
        cMid.Dispose(); xMid.Dispose();
        return (cOut, xOut);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] mods = [_modXW, _modXB, _modCW, _modCB];
        foreach (Tensor? t in mods) if (t is not null) yield return t;
        foreach (Tensor t in _attnX.EnumerateWeights()) yield return t;
        foreach (Tensor t in _attnC.EnumerateWeights()) yield return t;
        foreach (Tensor t in _ffX.EnumerateWeights()) yield return t;
        foreach (Tensor t in _ffC.EnumerateWeights()) yield return t;
    }

    private static Tensor Join(IBackend backend, TensorShape shape, Tensor audio, Tensor text)
    {
        Tensor joined = new(shape, DType.F32);
        backend.Concat(joined, [audio, text], 1);
        return joined;
    }
}
