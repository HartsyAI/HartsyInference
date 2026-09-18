using System.Reflection;
using System.Text.RegularExpressions;
using HartsyInference.Core.Configuration;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>A test may not configure the engine by setting an environment variable that a knob used to be read
/// from, because nothing reads the environment any more and the call silently does nothing.
///
/// <para>This is the check that would have caught the 28 red CUDA tests years earlier than a failing assertion did.
/// Worse than the failures were the tests that kept PASSING while configuring nothing: they were measuring whatever
/// the defaults happen to do, under a name that claims otherwise. A test that cannot fail for the reason it names is
/// the most expensive kind of green.</para>
///
/// <para>The list of forbidden names comes from the knob registry's own <see cref="Knob{T}.LegacyEnv"/>, not from a
/// hardcoded copy, so retiring or renaming a knob keeps this honest for free. Variables that are NOT knobs —
/// <c>HARTSY_REQUIRE_REAL_WEIGHTS</c>, <c>HARTSY_REQUIRE_BACKENDS</c>, <c>HARTSY_ALLOW_SOFTWARE_GPU</c>, the
/// <c>HARTSY_RUN_*</c> switches, checkpoint paths — are read by the test harness itself and stay perfectly
/// legitimate.</para></summary>
public sealed class TestsDoNotSetKnobEnvVarsTests
{
    private static readonly Regex SetEnv = new(
        @"SetEnvironmentVariable\(\s*""(?<name>[A-Z0-9_]+)""", RegexOptions.Compiled);

    /// <summary>Every legacy environment name the knob registry still answers to, with the setting that replaced it.</summary>
    private static Dictionary<string, string> LegacyNames()
    {
        Dictionary<string, string> names = new(StringComparer.Ordinal);
        foreach (FieldInfo field in typeof(EngineKnobs).GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            if (!field.FieldType.IsGenericType || field.FieldType.GetGenericTypeDefinition() != typeof(Knob<>))
            {
                continue;
            }
            object knob = field.GetValue(null)!;
            string? legacy = (string?)knob.GetType().GetProperty(nameof(Knob<int>.LegacyEnv))!.GetValue(knob);
            string id = (string)knob.GetType().GetProperty(nameof(Knob<int>.Id))!.GetValue(knob)!;
            if (!string.IsNullOrEmpty(legacy))
            {
                // Two knobs may share one legacy name; report both, because a caller has to set both.
                names[legacy] = names.TryGetValue(legacy, out string? already) ? $"{already} and {id}" : id;
            }
        }
        return names;
    }

    [Fact]
    public void No_Test_Configures_The_Engine_Through_A_Dead_Environment_Variable()
    {
        Dictionary<string, string> legacy = LegacyNames();
        Assert.NotEmpty(legacy);

        List<string> violations = [];
        string tests = Path.Combine(RepoRoot.Path, "tests");
        foreach (string file in Directory.EnumerateFiles(tests, "*.cs", SearchOption.AllDirectories))
        {
            string relative = Path.GetRelativePath(RepoRoot.Path, file).Replace('\\', '/');
            if (relative.Contains("/obj/", StringComparison.Ordinal) || relative.Contains("/bin/", StringComparison.Ordinal))
            {
                continue;
            }
            foreach (Match match in SetEnv.Matches(File.ReadAllText(file)))
            {
                string name = match.Groups["name"].Value;
                if (legacy.TryGetValue(name, out string? setting))
                {
                    violations.Add($"{relative}: {name} (use KnobStore.Set on {setting})");
                }
            }
        }

        Assert.True(violations.Count == 0,
            "These tests configure the engine through an environment variable nothing reads, so the call does "
            + "nothing and the test measures the default instead:\n  "
            + string.Join("\n  ", violations.Distinct().Order(StringComparer.Ordinal)));
    }
}
