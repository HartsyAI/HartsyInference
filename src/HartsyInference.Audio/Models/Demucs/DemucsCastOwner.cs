using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Demucs;

/// <summary>Load-time F32 conversion that records only the tensors it had to allocate, so a Demucs module can free
/// exactly its own casts and never touch the loader-owned tensors it merely borrowed.</summary>
internal sealed class DemucsCastOwner : IDisposable
{
    private readonly List<Tensor> _casts = [];

    /// <summary>F32 view of the weight at <paramref name="key"/>; the cast copy, if one was needed, is owned here.</summary>
    public Tensor F32(IReadOnlyDictionary<string, Tensor> w, string key) => F32(w[key]);

    /// <summary>As <see cref="F32(IReadOnlyDictionary{string, Tensor}, string)"/>, but null when the key is absent.</summary>
    public Tensor? Optional(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? t) ? F32(t) : null;

    private Tensor F32(Tensor t)
    {
        Tensor f = WhisperOps.EnsureF32(t);
        if (!ReferenceEquals(f, t)) _casts.Add(f);
        return f;
    }

    public void Dispose()
    {
        for (int i = 0; i < _casts.Count; i++) _casts[i].Dispose();
        _casts.Clear();
    }
}
