namespace HartsyInference.ModelAssets.Dcp;

/// <summary>An inert stand-in for a Python object in a DCP <c>.metadata</c> pickle: the class it names and the fields its state set. Nothing is imported or called.</summary>
internal sealed class DcpObject(string module, string name)
{
    public string Module { get; } = module;

    public string Name { get; } = name;

    /// <summary>Constructor arguments of a reduce call, when the object came from one.</summary>
    public object?[]? Args { get; set; }

    /// <summary>The attributes a BUILD opcode assigned.</summary>
    public Dictionary<string, object?> Fields { get; } = [];

    public T? Field<T>(string name) where T : class => Fields.TryGetValue(name, out object? value) ? value as T : null;
}
