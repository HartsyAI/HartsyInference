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


    /// <summary>Third-party and platform names the engine consumes but does not own. Not knobs.</summary>
    private static readonly HashSet<string> Foreign = new(StringComparer.Ordinal)
    {
        "HF_TOKEN", "HF_HOME", "HF_ENDPOINT", "HUGGINGFACE_HUB_TOKEN",
        "NO_COLOR", "COLORFGBG", "ESPEAK_DATA_DIR",
        "PATH", "HOME", "TEMP", "TMP", "USERPROFILE", "LD_LIBRARY_PATH",
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


    /// <summary>Every environment name the source still reads is either declared or explicitly deferred.</summary>
    /// <summary>No engine code reads the process environment, except the handful of third-party names we do not own.</summary>
    /// <remarks>This is the ratchet the environment removal left behind. It replaces the older
    /// "every name is declared or deferred" check, which could only ever be satisfied by recording a dead
    /// variable name on every knob — the surface this test now guarantees stays empty.</remarks>
    [Fact]
    public void NoEngineCodeReadsTheEnvironment()
    {
        List<string> offenders = [.. ScanSourceForEnvNames().Where(n => !Foreign.Contains(n)).Order()];

        Assert.True(offenders.Count == 0,
            "src/ reads these environment variables. The engine is configured through EngineKnobs and the settings\n"
            + "file; a second source of truth is what the removal was for:\n"
            + string.Join("\n", offenders.Select(n => "  " + n))
            + "\n\nDeclare a knob instead, or add the name to Foreign if it is a third-party convention we only honour.");
    }

    /// <summary>The two behaviours split across a PAIR of knobs still have both halves, with their opposite defaults.</summary>
    /// <remarks>Caught a real regression once: a generator rewrite dropped the second graph-capture knob while
    /// name-coverage stayed green, which would have quietly stopped the default-ON tier capturing graphs. The pins
    /// used to key off a shared environment name; they key off the ids now, which is what actually matters.</remarks>
    [Fact]
    public void DeliberateKnobPairsKeepBothHalves()
    {
        (string Id, bool Default)[] pairs =
        [
            // Graph capture: the opt-in tier and the default-ON tier. Changing one alone moves half the behaviour.
            ("numerics.ditGraph", false),
            ("numerics.ditGraphDefaultOn", true),
            // SageAttention: the default-ON kernel, and the explicit opt-in that ALONE unlocks the unsafe
            // F32 -> F16 V-narrowing. Collapsing these would open that path by default.
            ("numerics.sageAttn", true),
            ("numerics.sageAttnExplicit", false),
        ];

        List<string> problems = [];
        foreach ((string id, bool want) in pairs)
        {
            object? knob = KnobRegistry.Find(id);
            if (knob is null)
            {
                problems.Add($"  {id} is gone — half of a deliberate pair");
                continue;
            }
            object? got = KnobRegistry.Describe(knob).Default;
            if (got is not bool b || b != want)
            {
                problems.Add($"  {id} defaults to {got}, expected {want}");
            }
        }
        Assert.True(problems.Count == 0, "Deliberate knob pairs:\n" + string.Join("\n", problems));
    }

    /// <summary>Ids are unique, dotted, domain-prefixed, and carry no vendor prefix.</summary>
    [Fact]
    public void IdsAreWellFormed()
    {
        List<string> bad = [];
        foreach (object knob in KnobRegistry.All)
        {
            (string id, _, _, _, KnobDomain domain, string summary) = KnobRegistry.Describe(knob);
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

    /// <summary>Reports the declared surface by domain and scope, so a knob landing in the wrong bucket is visible in the test log.</summary>
    [Fact]
    public void ReportDeclaredSurface()
    {
        List<(string Id, string Type, object? Default, KnobScope Scope, KnobDomain Domain, string Summary)> real =
            [.. KnobRegistry.All.Select(KnobRegistry.Describe).Where(d => !IsTestKnob(d.Id))];
        _output.WriteLine($"{real.Count} knobs declared.");
        foreach (IGrouping<KnobDomain, (string Id, string Type, object? Default, KnobScope Scope, KnobDomain Domain, string Summary)> byDomain
                 in real.GroupBy(d => d.Domain).OrderBy(g => g.Key.ToString(), StringComparer.Ordinal))
        {
            string scopes = string.Join(", ", byDomain.GroupBy(d => d.Scope)
                .OrderBy(g => g.Key.ToString(), StringComparer.Ordinal)
                .Select(g => $"{g.Key} {g.Count()}"));
            _output.WriteLine($"  {byDomain.Key,-12} {byDomain.Count(),3}  ({scopes})");
        }
        Assert.True(real.Count > 0);
    }
}
