using System.Reflection;
using HartsyInference.Core.Configuration;
using HartsyInference.Tests.Common;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>A test may not configure the engine by naming an environment variable that a knob used to be read
/// from, because nothing reads the environment any more and the call silently does nothing.
///
/// <para>This is the check that would have caught the 28 red CUDA tests long before a failing assertion did. Worse
/// than the failures were the tests that kept PASSING while configuring nothing: they were measuring whatever the
/// defaults happen to do, under a name claiming otherwise. One suite ran both arms of a tiled-vs-untiled comparison
/// in the same configuration for exactly this reason. A test that cannot fail for the reason it names is the most
/// expensive kind of green.</para>
///
/// <para>It looks for the NAME as a string literal anywhere in the file rather than for a call to
/// <c>Environment.SetEnvironmentVariable</c>, because the call is the part that is easy to hide — behind a shared
/// helper, or behind a table of name/value pairs applied in a loop, which is how the suite above actually did it.
/// The name is the thing that cannot be hidden. A test that legitimately needs to talk about a setting should name
/// the dotted id, which is what <c>--set</c> and the settings file accept.</para>
///
/// <para>The forbidden list comes from the registry's own <see cref="Knob{T}.LegacyEnv"/>, not a hardcoded copy, so
/// retiring or renaming a knob keeps this honest for free. Variables that are NOT knobs —
/// <c>HARTSY_REQUIRE_REAL_WEIGHTS</c>, <c>HARTSY_REQUIRE_BACKENDS</c>, <c>HARTSY_ALLOW_SOFTWARE_GPU</c>, the
/// <c>HARTSY_RUN_*</c> switches, checkpoint paths — are read by the test harness itself and stay perfectly
/// legitimate.</para></summary>
public sealed class TestsDoNotSetKnobEnvVarsTests
{
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

    /// <summary>The two places a legacy name may legitimately appear as a literal.</summary>
    /// <remarks>Both are about the names rather than uses of them, which is the distinction the rule cares about.
    /// <see cref="Configuration.KnobRegistryTests"/> asserts on the registry's own contract — including that two
    /// knobs deliberately share one legacy name — and cannot do that without writing one down. The shared harness
    /// reads a real environment variable to locate the repo before any knob exists to consult; it happens to share a
    /// name with a knob, which is worth knowing but is not a configuration call.</remarks>
    private static readonly string[] MayNameThem =
    [
        "tests/HartsyInference.Core.Tests/Configuration/KnobRegistryTests.cs",
        "tests/HartsyInference.Tests.Common/RepoRoot.cs",
    ];

    [Fact]
    public void No_Test_Names_A_Dead_Environment_Variable()
    {
        Dictionary<string, string> legacy = LegacyNames();
        Assert.NotEmpty(legacy);

        List<string> violations = [];
        string tests = Path.Combine(RepoRoot.Path, "tests");
        foreach (string file in Directory.EnumerateFiles(tests, "*.cs", SearchOption.AllDirectories))
        {
            string relative = Path.GetRelativePath(RepoRoot.Path, file).Replace('\\', '/');
            if (relative.Contains("/obj/", StringComparison.Ordinal) || relative.Contains("/bin/", StringComparison.Ordinal)
                || MayNameThem.Contains(relative, StringComparer.Ordinal))
            {
                continue;
            }
            SyntaxNode root = CSharpSyntaxTree.ParseText(File.ReadAllText(file)).GetRoot();
            foreach (LiteralExpressionSyntax literal in root.DescendantNodes().OfType<LiteralExpressionSyntax>())
            {
                if (!literal.IsKind(SyntaxKind.StringLiteralExpression))
                {
                    continue;
                }
                if (legacy.TryGetValue(literal.Token.ValueText, out string? setting))
                {
                    violations.Add($"{relative}: \"{literal.Token.ValueText}\" (use KnobStore.Set on {setting})");
                }
            }
        }

        Assert.True(violations.Count == 0,
            "These tests name an environment variable nothing reads, so setting it does nothing and the test "
            + "measures the default instead:\n  "
            + string.Join("\n  ", violations.Distinct().Order(StringComparer.Ordinal)));
    }
}
