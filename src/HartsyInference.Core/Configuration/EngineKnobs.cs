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

    private static Knob<int> Int(string id, string? legacyEnv, int defaultValue, KnobScope scope, KnobDomain domain, string summary,
        Func<int, int>? coerce = null)
        => new(id, legacyEnv, defaultValue, scope, domain, summary, BoolGrammar.Exact, coerce);

    private static Knob<long> Long(string id, string? legacyEnv, long defaultValue, KnobScope scope, KnobDomain domain, string summary,
        Func<long, long>? coerce = null)
        => new(id, legacyEnv, defaultValue, scope, domain, summary, BoolGrammar.Exact, coerce);

    private static Knob<float> Float(string id, string? legacyEnv, float defaultValue, KnobScope scope, KnobDomain domain, string summary,
        Func<float, float>? coerce = null)
        => new(id, legacyEnv, defaultValue, scope, domain, summary, BoolGrammar.Exact, coerce);

    private static Knob<string?> Str(string id, string? legacyEnv, string? defaultValue, KnobScope scope, KnobDomain domain, string summary)
        => new(id, legacyEnv, defaultValue, scope, domain, summary);

    /// <summary>A knob whose default is not a constant — the call site computes it from hardware or model config.</summary>
    /// <remarks><c>null</c> means "no opinion": the caller keeps its own default. So a legacy tri-state read of
    /// <c>HARTSY_FP8_NATIVE</c> whose default argument was <c>fp8TensorCores</c> becomes
    /// <c>Fp8Native.Value ?? fp8TensorCores</c>, which is the same function of the same inputs.
    /// Without this shape the registry would have to bake a constant and would silently turn FP8 on for
    /// pre-Ada cards that cannot do it.</remarks>
    private static Knob<bool?> BoolOverride(string id, string? legacyEnv, KnobScope scope, KnobDomain domain, string summary)
        => new(id, legacyEnv, null, scope, domain, summary);

    private static Knob<int?> IntOverride(string id, string? legacyEnv, KnobScope scope, KnobDomain domain, string summary)
        => new(id, legacyEnv, null, scope, domain, summary);

    private static Knob<float?> FloatOverride(string id, string? legacyEnv, KnobScope scope, KnobDomain domain, string summary)
        => new(id, legacyEnv, null, scope, domain, summary);
}
