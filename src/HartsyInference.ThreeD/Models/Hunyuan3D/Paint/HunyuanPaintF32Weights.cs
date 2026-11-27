using System.Collections;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>A read-only view of a checkpoint that hands out every tensor as F32, converting on first use and keeping the copy; the source is untouched.
/// Disposing releases the converted copies (and their GPU residency) but never the source tensors.</summary>
internal sealed class HunyuanPaintF32Weights : IReadOnlyDictionary<string, Tensor>, IDisposable
{
    private readonly IReadOnlyDictionary<string, Tensor> _source;
    private readonly Dictionary<string, Tensor> _converted = [];

    internal HunyuanPaintF32Weights(IReadOnlyDictionary<string, Tensor> source) => _source = source;

    public Tensor this[string key]
    {
        get
        {
            if (!_converted.TryGetValue(key, out Tensor? tensor))
                _converted[key] = tensor = Hunyuan3DShapeVae.F32(_source[key]);
            return tensor;
        }
    }

    /// <summary>Releases every converted copy; tensors that passed through unconverted belong to the source and stay.</summary>
    public void Dispose()
    {
        foreach ((string key, Tensor tensor) in _converted)
            if (!ReferenceEquals(tensor, _source[key]))
                tensor.Dispose();
        _converted.Clear();
    }

    public IEnumerable<string> Keys => _source.Keys;
    public IEnumerable<Tensor> Values => _source.Keys.Select(k => this[k]);
    public int Count => _source.Count;
    public bool ContainsKey(string key) => _source.ContainsKey(key);

    public bool TryGetValue(string key, out Tensor value)
    {
        if (!_source.ContainsKey(key))
        {
            value = null!;
            return false;
        }
        value = this[key];
        return true;
    }

    public IEnumerator<KeyValuePair<string, Tensor>> GetEnumerator() => _source.Keys.Select(k => KeyValuePair.Create(k, this[k])).GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
