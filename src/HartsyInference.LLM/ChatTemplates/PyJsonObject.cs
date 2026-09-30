namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Insertion-ordered JSON object with Python-dict semantics: a repeated key keeps its first position and takes the last value.</summary>
internal sealed class PyJsonObject
{
    private readonly List<string> _keys = [];
    private readonly Dictionary<string, object?> _values = new(StringComparer.Ordinal);

    public IReadOnlyList<string> Keys => _keys;

    public int Count => _keys.Count;

    public bool ContainsKey(string key) => _values.ContainsKey(key);

    public object? this[string key]
    {
        get => _values[key];
        set
        {
            if (!_values.ContainsKey(key)) _keys.Add(key);
            _values[key] = value;
        }
    }

    public object? GetOrNull(string key) => _values.TryGetValue(key, out object? value) ? value : null;

    public bool Remove(string key)
    {
        if (!_values.Remove(key)) return false;
        _keys.Remove(key);
        return true;
    }

    public PyJsonObject Clone()
    {
        PyJsonObject copy = new();
        foreach (string key in _keys) copy[key] = _values[key];
        return copy;
    }
}
