namespace HartsyInference.Core.Configuration;

/// <summary>Every engine setting, declared once with its id, type, default, scope and domain.</summary>
/// <remarks>Declarations live in the <c>EngineKnobs.*.cs</c> partials, one per <see cref="KnobDomain"/>. Nothing
/// here reads the environment — <see cref="KnobStore"/> is the only file that does, so the C1 allowlist can be
/// driven to a single entry.
/// <para>Ids are dotted and vendor-free (<c>numerics.sageAttention</c>, not <c>HARTSY_SAGE_ATTN</c>); the legacy
/// environment name is recorded per knob and honored until C7 retires it.</para></remarks>
public static partial class EngineKnobs
{
    /// <summary>Forces the declaration partials' field initializers so <see cref="KnobRegistry"/> sees a complete surface.</summary>
    /// <remarks>Every declaration is a static field of this one class across its partials, so running its class constructor declares all of them.</remarks>
    public static void EnsureDeclared()
        => System.Runtime.CompilerServices.RuntimeHelpers.RunClassConstructor(typeof(EngineKnobs).TypeHandle);

    private static Knob<bool> Bool(string id, string? legacyEnv, bool defaultValue, BoolGrammar grammar,
        KnobScope scope, KnobDomain domain, string summary)
        => new(id, legacyEnv, defaultValue, scope, domain, summary, grammar);

    private static Knob<int> Int(string id, string? legacyEnv, int defaultValue, KnobScope scope, KnobDomain domain, string summary)
        => new(id, legacyEnv, defaultValue, scope, domain, summary);

    private static Knob<long> Long(string id, string? legacyEnv, long defaultValue, KnobScope scope, KnobDomain domain, string summary)
        => new(id, legacyEnv, defaultValue, scope, domain, summary);

    private static Knob<float> Float(string id, string? legacyEnv, float defaultValue, KnobScope scope, KnobDomain domain, string summary)
        => new(id, legacyEnv, defaultValue, scope, domain, summary);

    private static Knob<string?> Str(string id, string? legacyEnv, string? defaultValue, KnobScope scope, KnobDomain domain, string summary)
        => new(id, legacyEnv, defaultValue, scope, domain, summary);
}
