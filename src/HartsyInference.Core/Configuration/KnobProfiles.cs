namespace HartsyInference.Core.Configuration;

/// <summary>The named profiles an operator can select by name.</summary>
public static partial class KnobProfiles
{
    /// <summary>Every knob at its declared default. Selecting it is a no-op, which is the point — it names the baseline.</summary>
    public static readonly KnobProfile Default = KnobProfile.Create("default");

    /// <summary>Every approximation and fast path off, for parity work against a reference implementation.</summary>
    /// <remarks>Populated in <c>KnobProfiles.Reference.cs</c> from a per-knob audit of what each one actually does
    /// to the arithmetic. Only knobs with a clearly MORE faithful direction are pinned: a knob that merely selects
    /// which variant loads, or one a model requires, is left alone rather than forced, since pinning those would
    /// break a model rather than make it faithful.
    /// <para>This is not a promise of bit-exactness with any particular reference — it is the most faithful
    /// configuration the engine can be put into from settings alone.</para></remarks>
    public static KnobProfile Reference => ReferenceProfile;

    public static IEnumerable<string> Names => ["default", "reference"];

    /// <summary>Resolves a profile by the name an operator typed. Null when unknown, so the caller can list the valid ones.</summary>
    public static KnobProfile? ByName(string name) => name.Trim().ToLowerInvariant() switch
    {
        "default" => Default,
        "reference" => Reference,
        _ => null,
    };
}
