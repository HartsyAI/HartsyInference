using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Weight lookup for the LTX-2.5 diffusion decoder tree. GEMM weights stay in their stored dtype (the Linear
/// kernels cast them); norm scales and biases are read as raw <c>float*</c> by their kernels and must be promoted.
/// Tracks every key it hands out so a load can be checked for unconsumed checkpoint entries, and owns the cast copies
/// so the decoder can free them.</summary>
internal sealed class LtxVideo25WeightScope
{
    private readonly IReadOnlyDictionary<string, Tensor> _weights;
    private readonly HashSet<string> _consumed = [];
    private readonly List<Tensor> _owned = [];

    public LtxVideo25WeightScope(IReadOnlyDictionary<string, Tensor> weights) => _weights = weights;

    /// <summary>Keys read so far, in checkpoint naming.</summary>
    public IReadOnlyCollection<string> Consumed => _consumed;

    /// <summary>Cast copies this scope allocated; the decoder disposes them.</summary>
    public IReadOnlyList<Tensor> Owned => _owned;

    /// <summary>Weight as stored, naming the key when it is absent.</summary>
    public Tensor Raw(string key)
    {
        if (!_weights.TryGetValue(key, out Tensor? t))
            throw new KeyNotFoundException($"LTX-2.5 diffusion decoder weight '{key}' is missing.");
        _consumed.Add(key);
        return t;
    }

    /// <summary>F32 view of a required weight.</summary>
    public Tensor F32(string key) => Promote(Raw(key));

    /// <summary>F32 view of an optional weight (the SwiGLU projections ship without biases).</summary>
    public Tensor? OptionalF32(string key)
    {
        if (!_weights.TryGetValue(key, out Tensor? t)) return null;
        _consumed.Add(key);
        return Promote(t);
    }

    /// <summary>F32 copy scaled by <paramref name="scale"/>. Always allocates: the source tensor is the loader's and
    /// is shared with anything else reading the checkpoint, so it must never be scaled in place.</summary>
    public unsafe Tensor ScaledF32(string key, float scale)
    {
        Tensor source = F32(key);
        Tensor scaled = new Tensor(source.Shape, DType.F32);
        float* src = (float*)source.DataPointer, dst = (float*)scaled.DataPointer;
        for (long i = 0; i < source.ElementCount; i++) dst[i] = src[i] * scale;
        _owned.Add(scaled);
        return scaled;
    }

    private Tensor Promote(Tensor t)
    {
        if (t.DType == DType.F32) return t;
        Tensor cast = t.CastTo(DType.F32);
        _owned.Add(cast);
        return cast;
    }
}
