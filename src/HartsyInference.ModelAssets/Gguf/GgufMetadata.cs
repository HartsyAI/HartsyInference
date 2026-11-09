namespace HartsyInference.ModelAssets.Gguf;

/// <summary>Typed metadata key-value access for GGUF file headers.</summary>
public sealed class GgufMetadata
{
    private readonly Dictionary<string, object> _values = [];

    /// <summary>All metadata keys present in the file.</summary>
    public IReadOnlyCollection<string> Keys => _values.Keys;

    /// <summary>Number of metadata entries.</summary>
    public int Count => _values.Count;

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

    /// <summary>Gets an integer metadata value stored as any GGUF integer type (converters differ: one writes uint32, another
    /// int32), or the default when absent or not an integer.</summary>
    /// <exception cref="OverflowException">The stored value does not fit an int.</exception>
    public int GetInteger(string key, int defaultValue = 0) =>
        _values.TryGetValue(key, out object? value) && value is byte or sbyte or short or ushort or int or uint or long or ulong
            ? Convert.ToInt32(value)
            : defaultValue;

    /// <summary>Gets an integer array stored as any GGUF integer element type, or null when absent or not an integer array.</summary>
    public int[]? GetIntegerArray(string key)
    {
        if (!_values.TryGetValue(key, out object? value)) return null;
        return value switch
        {
            int[] ints => ints,
            uint[] uints => uints.Select(u => checked((int)u)).ToArray(),
            long[] longs => longs.Select(l => checked((int)l)).ToArray(),
            ulong[] ulongs => ulongs.Select(u => checked((int)u)).ToArray(),
            short[] shorts => shorts.Select(s => (int)s).ToArray(),
            ushort[] ushorts => ushorts.Select(s => (int)s).ToArray(),
            _ => null,
        };
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

    public bool ContainsKey(string key) => _values.ContainsKey(key);

    /// <summary>Gets the raw value for a metadata key.</summary>
    public bool TryGetValue(string key, out object? value) => _values.TryGetValue(key, out value);
}
