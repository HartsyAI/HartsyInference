using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>One <c>LTX2AdaLayerNormSingle</c>: a PixArt timestep embedder (<c>emb.timestep_embedder.linear_1/2</c>:
/// sinusoidal-256 → SiLU → <c>dim</c>) feeding <c>linear</c> (SiLU → <c>numParams·dim</c>). Returns the flat
/// <c>[numParams·dim]</c> modulation vector and the <c>[dim]</c> embedded timestep (used by the output layer).</summary>
internal sealed class LtxVideo2AdaLnSingle
{
    private readonly int _dim, _numParams;
    private Tensor? _emb1W, _emb1B, _emb2W, _emb2B, _linW, _linB;

    public LtxVideo2AdaLnSingle(int dim, int numParams)
    {
        _dim = dim;
        _numParams = numParams;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _emb1W = w[$"{p}.emb.timestep_embedder.linear_1.weight"]; w.TryGetValue($"{p}.emb.timestep_embedder.linear_1.bias", out _emb1B);
        _emb2W = w[$"{p}.emb.timestep_embedder.linear_2.weight"]; w.TryGetValue($"{p}.emb.timestep_embedder.linear_2.bias", out _emb2B);
        _linW = w[$"{p}.linear.weight"]; w.TryGetValue($"{p}.linear.bias", out _linB);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _emb1W, _emb1B, _emb2W, _emb2B, _linW, _linB })
            if (t is not null) yield return t;
    }

    public (Tensor Mod, Tensor Embedded) Forward(IBackend backend, float timestep)
    {
        Tensor sin = new Tensor(new TensorShape(1, 256), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sin, timestep, 1, 256, 10000f);
        Tensor e1 = new Tensor(new TensorShape(1, _dim), DType.F32);
        backend.Linear(e1, sin, _emb1W!, _emb1B); sin.Dispose();
        Tensor e1a = new Tensor(e1.Shape, DType.F32); backend.Silu(e1a, e1); e1.Dispose();
        Tensor embedded2d = new Tensor(new TensorShape(1, _dim), DType.F32);
        backend.Linear(embedded2d, e1a, _emb2W!, _emb2B); e1a.Dispose();

        Tensor sil = new Tensor(new TensorShape(1, _dim), DType.F32); backend.Silu(sil, embedded2d);
        Tensor modFlat = new Tensor(new TensorShape(1, _numParams * _dim), DType.F32);
        backend.Linear(modFlat, sil, _linW!, _linB); sil.Dispose();

        // Returned as the [1, n·dim]/[1, dim] Linear outputs directly — every consumer is an element-count
        // device op, so a host repack here would D2H-drain both tensors per modulator per forward (the
        // Kandinsky temb stream-drain pathology).
        return (modFlat, embedded2d);
    }
}
