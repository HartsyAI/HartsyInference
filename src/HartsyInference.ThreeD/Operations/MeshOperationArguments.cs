namespace HartsyInference.ThreeD.Operations;

/// <summary>The validated numbers of one operation step: the request's values with the defaults filled in.</summary>
public sealed class MeshOperationArguments
{
    private readonly IReadOnlyDictionary<string, double> _values;

    internal MeshOperationArguments(IReadOnlyDictionary<string, double> values) => _values = values;

    /// <summary>Whether the value was given or has a default.</summary>
    public bool Has(string name) => _values.ContainsKey(name);

    /// <summary>The value of <paramref name="name"/>.</summary>
    /// <exception cref="KeyNotFoundException">It was neither given nor defaulted (the operation checks <see cref="Has"/> for the ones it makes optional).</exception>
    public double Get(string name) => _values[name];

    /// <summary>The value of <paramref name="name"/> as a whole number.</summary>
    public int GetInt(string name) => (int)_values[name];
}
