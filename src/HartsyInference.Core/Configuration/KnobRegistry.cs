using System.Collections.Concurrent;

namespace HartsyInference.Core.Configuration;

/// <summary>Every declared knob, so the CLI can list them, the API can validate a name, and tests can assert the surface.</summary>
public static class KnobRegistry
{
    private static readonly ConcurrentDictionary<string, object> _all = new(StringComparer.Ordinal);

    internal static void Register<T>(Knob<T> knob)
    {
        if (!_all.TryAdd(knob.Id, knob))
        {
            throw new InvalidOperationException($"Duplicate knob id '{knob.Id}'. Ids must be unique.");
        }
    }

    /// <summary>All declared knobs. Forces the declaration classes to run their static initializers first.</summary>
    public static IReadOnlyCollection<object> All
    {
        get
        {
            EngineKnobs.EnsureDeclared();
            return [.. _all.Values];
        }
    }

    /// <summary>Looks a knob up by dotted id, e.g. for CLI <c>--set numerics.sageAttention=0</c>.</summary>
    public static object? Find(string id)
    {
        EngineKnobs.EnsureDeclared();
        return _all.TryGetValue(id, out object? k) ? k : null;
    }

    /// <summary>Describes a knob for <c>--list-knobs</c> output.</summary>
    public static (string Id, string? LegacyEnv, string Type, object? Default, KnobScope Scope, KnobDomain Domain, string Summary) Describe(object knob)
    {
        Type t = knob.GetType();
        Type arg = t.GetGenericArguments()[0];
        object? Get(string n) => t.GetProperty(n)!.GetValue(knob);
        return ((string)Get("Id")!, (string?)Get("LegacyEnv"), arg.Name, Get("Default"),
            (KnobScope)Get("Scope")!, (KnobDomain)Get("Domain")!, (string)Get("Summary")!);
    }
}
