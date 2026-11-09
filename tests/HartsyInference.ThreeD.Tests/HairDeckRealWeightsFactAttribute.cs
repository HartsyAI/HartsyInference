namespace HartsyInference.ThreeD.Tests;

/// <summary>A <see cref="FactAttribute"/> reported as <i>skipped</i> (never passed) when a HairDeck real-weight input is
/// missing. With <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> it is not skipped: the test runs and fails on the missing input.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class HairDeckRealWeightsFactAttribute : FactAttribute
{
    public HairDeckRealWeightsFactAttribute()
    {
        if (Environment.GetEnvironmentVariable(HairDeckReferenceLayout.RequireEnv) == "1")
            return;
        string? missing = HairDeckReferenceLayout.MissingInputs().FirstOrDefault();
        if (missing is not null)
            Skip = $"SKIPPED: HairDeck real-weight input missing: {missing}; parity did not run.";
    }
}
