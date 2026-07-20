namespace HartsyInference.ModelAssets.Gguf;

/// <summary>Typed metadata key-value access for GGUF file headers.</summary>
public sealed class GgufMetadata
{
    private readonly Dictionary<string, object> _values = [];

    /// <summary>All metadata keys present in the file.</summary>
    public IReadOnlyCollection<string> Keys => _values.Keys;

    /// <summary>Number of metadata entries.</summary>
    public int Count => _values.Count;

    /// <summary>Adds a metadata key-value pair.</summary>
    internal void Add(string key, object value)
    {
        _values[key] = value;
    }

    /// <summary>Gets a string metadata value, or null if not found.</summary>
    public string? GetString(string key)
    {
        if (_values.TryGetValue(key, out object? value) && value is string s)
            return s;
        return null;
    }

    /// <summary>Gets a uint32 metadata value, or the default if not found.</summary>
    public uint GetUInt32(string key, uint defaultValue = 0)
    {
        if (_values.TryGetValue(key, out object? value) && value is uint u)
            return u;
        return defaultValue;
    }

    /// <summary>Gets an int32 metadata value, or the default if not found.</summary>
    public int GetInt32(string key, int defaultValue = 0)
    {
        if (_values.TryGetValue(key, out object? value) && value is int i)
            return i;
        return defaultValue;
    }

    /// <summary>Gets a float32 metadata value, or the default if not found.</summary>
    public float GetFloat32(string key, float defaultValue = 0f)
    {
        if (_values.TryGetValue(key, out object? value) && value is float f)
            return f;
        return defaultValue;
    }

    /// <summary>Gets a uint64 metadata value, or the default if not found.</summary>
    public ulong GetUInt64(string key, ulong defaultValue = 0)
    {
        if (_values.TryGetValue(key, out object? value) && value is ulong u)
            return u;
        return defaultValue;
    }

    /// <summary>Gets a bool metadata value, or the default if not found.</summary>
    public bool GetBool(string key, bool defaultValue = false)
    {
        if (_values.TryGetValue(key, out object? value) && value is bool b)
            return b;
        return defaultValue;
    }

    /// <summary>Gets a string array metadata value, or null if not found.</summary>
    public string[]? GetStringArray(string key)
    {
        if (_values.TryGetValue(key, out object? value) && value is string[] arr)
            return arr;
        return null;
    }

    /// <summary>Gets an int32/uint32 array metadata value (e.g. <c>tokenizer.ggml.token_type</c>), or null.</summary>
    public int[]? GetIntArray(string key)
    {
        if (_values.TryGetValue(key, out object? value) && value is int[] arr)
            return arr;
        return null;
    }

    /// <summary>Gets a float32 array metadata value (e.g. <c>tokenizer.ggml.scores</c> for SentencePiece), or null.</summary>
    public float[]? GetFloatArray(string key)
    {
        if (_values.TryGetValue(key, out object? value) && value is float[] arr)
            return arr;
        return null;
    }

    /// <summary>Gets a bool array metadata value (e.g. Gemma-4's per-layer <c>attention.sliding_window_pattern</c>), or null.</summary>
    public bool[]? GetBoolArray(string key)
    {
        if (_values.TryGetValue(key, out object? value) && value is bool[] arr)
            return arr;
        return null;
    }

    /// <summary>Checks if a metadata key exists.</summary>
    public bool ContainsKey(string key) => _values.ContainsKey(key);

    /// <summary>Gets the raw value for a metadata key.</summary>
    public bool TryGetValue(string key, out object? value) => _values.TryGetValue(key, out value);
}
