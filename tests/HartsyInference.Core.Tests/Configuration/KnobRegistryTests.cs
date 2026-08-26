using System.Text.RegularExpressions;
using HartsyInference.Core.Configuration;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Configuration;

/// <summary>Ties the declared knob surface to the environment names the source actually reads, so a knob cannot be dropped on the way into the registry.</summary>
/// <remarks>The generator that produced most declarations worked from a source scan, and an earlier revision of it
/// silently skipped an identifier collision. A count that lives only in a build script proves nothing after the
/// script is deleted — this asserts the same property from the repo, every run.</remarks>
public sealed class KnobRegistryTests
{
    private readonly ITestOutputHelper _output;

    public KnobRegistryTests(ITestOutputHelper output) => _output = output;

    /// <summary>Knobs deliberately NOT in the registry yet, with the reason. Emptied by C3.</summary>
    /// <remarks>All five are the step-cache family. Their resolution throws on malformed input — deliberate, so a
    /// silently-ignored perf knob cannot invalidate an A/B run — and <c>HARTSY_STEP_CACHE</c> resolves against the
    /// pipeline's <c>StepCacheProfile</c>, which a nullary <c>Knob.Value</c> cannot express. <c>StepCacheEnv</c> is
    /// already the single reader for the family, which is the shape C3 wants anyway.</remarks>
    private static readonly HashSet<string> Deferred = new(StringComparer.Ordinal)
    {
        "HARTSY_STEP_CACHE",
        "HARTSY_STEP_CACHE_CAP",
        "HARTSY_STEP_CACHE_LATE",
        "HARTSY_STEP_CACHE_POLY",
        "HARTSY_STEP_CACHE_CALIB",
    };

    /// <summary>Third-party and platform names the engine consumes but does not own. Not knobs.</summary>
    private static readonly HashSet<string> Foreign = new(StringComparer.Ordinal)
    {
        "HF_TOKEN", "HF_HOME", "HF_ENDPOINT", "HUGGINGFACE_HUB_TOKEN",
        "NO_COLOR", "COLORFGBG", "ESPEAK_DATA_DIR",
        "PATH", "HOME", "TEMP", "TMP", "USERPROFILE", "LD_LIBRARY_PATH",
    };

    /// <summary>Engine knobs that exist but are not declared yet, with why. Emptied as the migration proceeds.</summary>
    /// <remarks>Found while piloting the Video package: the first inventory only matched <c>HARTSY_</c>/<c>HM_</c>
    /// prefixes, so these were invisible to it. They are ordinary engine knobs that happen to be named after their
    /// model or subsystem instead. Listed rather than quietly excluded so the registry cannot claim a completeness
    /// it does not have.</remarks>
    private static readonly HashSet<string> NotYetDeclared = new(StringComparer.Ordinal)
    {
        // Three-state on/off/auto grammar with its own logging, like the step-cache family. C3.
        "HARTSY_ANIMATE2_BF16_DRIVING_CACHE",
    };

    /// <summary>Names already absorbed into <c>VramPolicy</c>, where the environment read is the documented lowest-precedence fallback rather than a knob to declare.</summary>
    private static readonly HashSet<string> Absorbed = new(StringComparer.Ordinal)
    {
        "HARTSY_LOWVRAM", "HARTSY_KEEP_MODELS",
    };

    /// <remarks>Three forms, because a literal-only scan of <c>GetEnvironmentVariable</c> understated the surface
    /// badly. It missed every name reached through the <c>EnvFlag</c> helper (a whole family of GEMM and SDPA
    /// flags) and every name held in a <c>const</c> and passed by reference. Both were invisible to the inventory
    /// that produced the generated declarations.</remarks>
    private static readonly Regex[] EnvNamePatterns =
    [
        new(@"(?:Environment\.GetEnvironmentVariable|EnvSwitch\.(?:IsEnabled|GetInt|GetFloat|GetLong))\s*\(\s*""([A-Za-z_][A-Za-z0-9_]*)""", RegexOptions.Compiled),
        new(@"EnvFlag\s*\(\s*""([A-Z0-9_]+)""", RegexOptions.Compiled),
        new(@"const\s+string\s+\w+\s*=\s*""((?:HARTSY|HM)_[A-Z0-9_]+)""", RegexOptions.Compiled),
        // Names passed to a helper's constructor or method, e.g. new DebugDumpSink("WAN_DEBUG_DIR").
        new(@"new\s+DebugDumpSink\s*\(\s*""([A-Z0-9_]+)""", RegexOptions.Compiled),
        // Names that only ever appear as a default parameter value, e.g. FromEnvironment(string v = "HARTSY_CFG_INTERVAL").
        new(@"string\s+\w+\s*=\s*""((?:HARTSY|HM)_[A-Z0-9_]+)""", RegexOptions.Compiled),
    ];

    private static HashSet<string> ScanSourceForEnvNames()
    {
        HashSet<string> found = new(StringComparer.Ordinal);
        string src = Path.Combine(RepoRoot.Path, "src");
        foreach (string file in Directory.EnumerateFiles(src, "*.cs", SearchOption.AllDirectories))
        {
            string rel = file.Replace('\\', '/');
            if (rel.Contains("/obj/", StringComparison.Ordinal) || rel.Contains("/bin/", StringComparison.Ordinal))
            {
                continue;
            }
            string text = File.ReadAllText(file);
            foreach (Regex pattern in EnvNamePatterns)
            {
                foreach (Match m in pattern.Matches(text))
                {
                    found.Add(m.Groups[1].Value);
                }
            }
        }
        return found;
    }

    /// <summary>Ids declared by the tests themselves, which must not count toward the real surface.</summary>
    private static bool IsTestKnob(string id) => id.StartsWith("test.", StringComparison.Ordinal);

    private static Dictionary<string, List<string>> DeclaredByLegacyName()
    {
        Dictionary<string, List<string>> byEnv = new(StringComparer.Ordinal);
        foreach (object knob in KnobRegistry.All)
        {
            (string id, string? legacy, _, _, _, _, _) = KnobRegistry.Describe(knob);
            if (IsTestKnob(id) || legacy is null)
            {
                continue;
            }
            (byEnv.TryGetValue(legacy, out List<string>? ids) ? ids : byEnv[legacy] = []).Add(id);
        }
        return byEnv;
    }

    /// <summary>Every environment name the source still reads is either declared or explicitly deferred.</summary>
    [Fact]
    public void EveryEnvironmentNameIsDeclaredOrDeferred()
    {
        HashSet<string> inSource = ScanSourceForEnvNames();
        Dictionary<string, List<string>> declared = DeclaredByLegacyName();

        List<string> undeclared = [.. inSource
            .Where(n => !declared.ContainsKey(n) && !Deferred.Contains(n)
                     && !Foreign.Contains(n) && !NotYetDeclared.Contains(n) && !Absorbed.Contains(n))
            .Order()];

        Assert.True(undeclared.Count == 0,
            "These environment names are read by src/ but are neither declared in EngineKnobs nor listed:\n"
            + string.Join("\n", undeclared.Select(n => "  " + n))
            + "\n\nDeclare them in EngineKnobs, or add them to Deferred / NotYetDeclared / Foreign with the reason.");
    }

    /// <summary>A deferred entry that nothing reads any more must be deleted, so the list cannot hold stale exemptions.</summary>
    [Fact]
    public void DeferredListHasNoStaleEntries()
    {
        HashSet<string> inSource = ScanSourceForEnvNames();
        List<string> stale = [.. Deferred.Concat(NotYetDeclared).Concat(Absorbed).Where(n => !inSource.Contains(n)).Order()];

        Assert.True(stale.Count == 0,
            "These names are deferred or backlogged but nothing reads them any more — delete the entries:\n"
            + string.Join("\n", stale.Select(n => "  " + n)));
    }

    /// <summary>One legacy name backing several knobs is deliberate and enumerated here, so collapsing a pair is a failure rather than a silent default change.</summary>
    /// <remarks>Caught a real regression: a generator rewrite dropped the second <c>HARTSY_DIT_GRAPH</c> knob, and
    /// <see cref="EveryEnvironmentNameIsDeclaredOrDeferred"/> stayed green because the NAME was still covered. The
    /// surviving knob defaults to false, so the default-ON tier would have quietly stopped capturing graphs.</remarks>
    [Fact]
    public void SharedLegacyNamesAreExactlyTheIntendedOnes()
    {
        Dictionary<string, int> expected = new(StringComparer.Ordinal)
        {
            // Enabled (default OFF) and EnabledDefaultOn (default ON): =0 kills both, =1 forces both.
            ["HARTSY_DIT_GRAPH"] = 2,
        };

        Dictionary<string, List<string>> declared = DeclaredByLegacyName();
        List<string> problems = [];
        foreach ((string env, List<string> ids) in declared.Where(kv => kv.Value.Count > 1))
        {
            if (!expected.TryGetValue(env, out int want))
            {
                problems.Add($"  {env} unexpectedly backs {ids.Count} knobs: {string.Join(", ", ids)}");
            }
            else if (ids.Count != want)
            {
                problems.Add($"  {env} backs {ids.Count} knobs, expected {want}: {string.Join(", ", ids)}");
            }
        }
        foreach ((string env, int want) in expected)
        {
            int got = declared.TryGetValue(env, out List<string>? ids) ? ids.Count : 0;
            if (got != want)
            {
                problems.Add($"  {env} backs {got} knobs, expected {want} — a deliberate multi-knob name was collapsed");
            }
        }
        Assert.True(problems.Count == 0, "Legacy names backing multiple knobs:\n" + string.Join("\n", problems));
    }

    /// <summary>Ids are unique, dotted, domain-prefixed, and carry no vendor prefix.</summary>
    [Fact]
    public void IdsAreWellFormed()
    {
        List<string> bad = [];
        foreach (object knob in KnobRegistry.All)
        {
            (string id, _, _, _, _, KnobDomain domain, string summary) = KnobRegistry.Describe(knob);
            if (IsTestKnob(id))
            {
                continue;
            }
            string expected = domain switch
            {
                KnobDomain.Numerics => "numerics.",
                KnobDomain.Vram => "vram.",
                KnobDomain.Diagnostics => "diagnostics.",
                KnobDomain.Paths => "paths.",
                _ => "?",
            };
            if (!id.StartsWith(expected, StringComparison.Ordinal))
            {
                bad.Add($"  {id}: domain {domain} expects prefix '{expected}'");
            }
            if (id.Contains("hartsy", StringComparison.OrdinalIgnoreCase))
            {
                bad.Add($"  {id}: ids carry no vendor prefix");
            }
            if (string.IsNullOrWhiteSpace(summary))
            {
                bad.Add($"  {id}: needs a summary");
            }
        }
        Assert.True(bad.Count == 0, "Malformed knob ids:\n" + string.Join("\n", bad));
    }

    /// <summary>Reports the declared surface, so the migration's progress is visible in the test log.</summary>
    [Fact]
    public void ReportDeclaredSurface()
    {
        Dictionary<string, List<string>> declared = DeclaredByLegacyName();
        int real = KnobRegistry.All.Count(k => !IsTestKnob(KnobRegistry.Describe(k).Id));
        _output.WriteLine($"{real} knobs declared, covering {declared.Count} legacy environment names; {Deferred.Count} deferred.");
        foreach ((string env, List<string> ids) in declared.Where(kv => kv.Value.Count > 1))
        {
            _output.WriteLine($"  {env} backs {ids.Count} knobs: {string.Join(", ", ids)}");
        }
        Assert.True(real > 0);
    }
}
