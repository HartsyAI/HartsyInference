using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Auk;

/// <summary>AuK single-stream DiT block over the concatenated [text | audio] sequence.</summary>
public sealed class AukSingleBlock(AukConfig config)
{
    private readonly AukAttnProj _attn = new();
    private readonly AukSwiGluFfn _ff = new();
    private Tensor? _modW, _modB;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        int dim = config.Dim;
        _modW = AukOps.Take(weights, $"{prefix}.attn_norm.linear.weight", 6 * dim, dim);
        _modB = AukOps.Take(weights, $"{prefix}.attn_norm.linear.bias", 6 * dim);
        _attn.Load(weights, $"{prefix}.attn", "to_qkv", "q_norm", "k_norm", "to_out.0", dim, config.Heads);
        _ff.Load(weights, $"{prefix}.ff", dim, config.FfInner);
    }

    /// <summary>Returns a new <c>[1, n, dim]</c> tensor; <paramref name="x"/> stays with the caller and RoPE positions run 0..n-1 of the tables.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor siluTime, int n, Tensor cos, Tensor sin)
    {
        int dim = config.Dim;
        using AukModulation mod = new(backend, siluTime, _modW!, _modB!, dim);
        Tensor modded = AukOps.AdaNorm(backend, x, mod.ScaleMsa, mod.ShiftMsa, config.NormEps);
        (Tensor q, Tensor k, Tensor v) = _attn.Project(backend, modded, n, cos, sin);
        modded.Dispose();
        Tensor merged = AukOps.Attend(backend, q, k, v, n, config.Heads, config.HeadDim);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor attnOut = _attn.Output(backend, merged, n);
        merged.Dispose();
        Tensor afterAttn = AukOps.GatedResidual(backend, x, attnOut, mod.GateMsa);
        attnOut.Dispose();
        Tensor result = _ff.ForwardResidual(backend, afterAttn, mod, n, config.NormEps);
        afterAttn.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_modW is not null) yield return _modW;
        if (_modB is not null) yield return _modB;
        foreach (Tensor t in _attn.EnumerateWeights()) yield return t;
        foreach (Tensor t in _ff.EnumerateWeights()) yield return t;
    }
}
