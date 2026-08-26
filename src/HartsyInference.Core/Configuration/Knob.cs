namespace HartsyInference.Core.Configuration;

/// <summary>When a knob binds: <see cref="Runtime"/> is re-read per generation, <see cref="Construction"/> is baked into a model at load and changing it requires a pipeline rebuild.</summary>
public enum KnobScope
{
    /// <summary>Read each generation; safe to override per request.</summary>
    Runtime,

    /// <summary>Baked at load time. Overriding per request would need the value in the recipe cache key.</summary>
    Construction,
}

/// <summary>Which typed options record a knob will fold into in C3.</summary>
public enum KnobDomain
{
    /// <summary>Kernel selection, precision, fusion, sampler math.</summary>
    Numerics,

    /// <summary>Residency, streaming, chunking, cache precision.</summary>
    Vram,

    /// <summary>Tracing, dumps, probes, profiling. Never affects output.</summary>
    Diagnostics,

    /// <summary>Filesystem locations for native libraries and assets.</summary>
    Paths,
}

/// <summary>One declared engine setting: its id, type, default, scope and domain, resolved through <see cref="KnobStore"/>.</summary>
/// <remarks>Declaring a knob does not read anything — resolution happens on <see cref="Value"/>. Call sites that
/// cache into a <c>static readonly</c> field keep exactly the binding time they had before the migration.
/// <para>Two knobs may share a <paramref name="legacyEnv"/> name with different defaults; <c>HARTSY_DIT_GRAPH</c>
/// deliberately drives both an opt-in and a default-on flag so <c>=0</c> kills both and <c>=1</c> forces both.</para></remarks>
public sealed class Knob<T>
{
    internal Knob(string id, string? legacyEnv, T defaultValue, KnobScope scope, KnobDomain domain, string summary,
        BoolGrammar grammar = BoolGrammar.Exact, Func<T, T>? coerce = null)
    {
        Id = id;
        LegacyEnv = legacyEnv;
        Default = defaultValue;
        Scope = scope;
        Domain = domain;
        Summary = summary;
        Grammar = grammar;
        Coerce = coerce;
        KnobRegistry.Register(this);
    }

    /// <summary>Only meaningful when <typeparamref name="T"/> is <see cref="bool"/>; see <see cref="BoolGrammar"/>.</summary>
    internal BoolGrammar Grammar { get; }

    /// <summary>Applied to the parsed value to produce the effective one. Null means take it as parsed.</summary>
    /// <remarks>Load-bearing, not decorative — the historic call sites did two different things with an
    /// out-of-range value and both must survive verbatim. Some rejected it
    /// (<c>TryParse(..) &amp;&amp; gb &gt; 0 ? gb : 14</c>), so a bare parse would newly ACCEPT negatives; others
    /// clamped it (<c>Math.Clamp(w, 1, 16)</c>), where <c>17</c> means 16 and emphatically not the default 4.
    /// One coercion function expresses both, so the registry does not grow a second grammar to choose between.</remarks>
    internal Func<T, T>? Coerce { get; }

    /// <summary>Dotted id, e.g. <c>numerics.sageAttention</c>. The name used by CLI <c>--set</c> and the API.</summary>
    public string Id { get; }

    /// <summary>Environment name this knob used to be read from, honored during the deprecation window. Null once retired.</summary>
    public string? LegacyEnv { get; }

    public T Default { get; }

    public KnobScope Scope { get; }

    public KnobDomain Domain { get; }

    /// <summary>One line describing what the knob does, surfaced by <c>--list-knobs</c> and the docs generator.</summary>
    public string Summary { get; }

    /// <summary>Resolved value: explicit override first, then the legacy environment name, then <see cref="Default"/>.</summary>
    public T Value => KnobStore.Resolve(this);

    public override string ToString() => Id;
}
