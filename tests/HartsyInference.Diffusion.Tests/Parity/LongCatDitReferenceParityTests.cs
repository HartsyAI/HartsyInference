using Xunit;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>Every stage of the tiny LongCat DiT against the pinned reference modules
/// (tests/python-reference/longcat_dit/dump_longcat_dit.py), CPU F32.
/// Skipped, and so not evidence, until HARTSY_LONGCAT_DIT_DIR names the dump directory.</summary>
[Trait("Category", "Integration")]
public sealed class LongCatDitReferenceParityTests
{
    private const string EnvName = "HARTSY_LONGCAT_DIT_DIR";

    [LongCatReferenceFact]
    public void EveryStageMatchesTheReference()
    {
        string dir = Environment.GetEnvironmentVariable(EnvName)!;
        (List<string> failures, List<string> passed) = LongCatReferenceDump.Compare(dir);
        File.WriteAllLines(Path.Combine(dir, "parity-report.txt"),
            failures.Select(static l => "FAIL " + l).Concat(passed.Select(static l => "ok   " + l)));

        Assert.True(failures.Count == 0, $"{failures.Count} stage(s) off, {passed.Count} matched:\n{string.Join('\n', failures)}");
    }

    [AttributeUsage(AttributeTargets.Method)]
    private sealed class LongCatReferenceFactAttribute : FactAttribute
    {
        public LongCatReferenceFactAttribute()
        {
            string? dir = Environment.GetEnvironmentVariable(EnvName);
            if (string.IsNullOrWhiteSpace(dir) || !File.Exists(Path.Combine(dir, "meta.json")))
                Skip = $"SKIPPED: {EnvName} is unset or has no meta.json; the LongCat DiT reference comparison did not run.";
        }
    }
}
