using System.Reflection;
using System.Text.RegularExpressions;
using HartsyInference.Core.Configuration;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>Makes <see cref="KnobScope"/> mean something. A <see cref="KnobScope.Runtime"/> knob is declared to be
/// "read each generation; safe to override per request" — so caching one into a <c>static readonly</c> field, which
/// freezes it at type-initialization for the life of the process, is a violation of its own declaration.
///
/// <para>This was not hypothetical. 53 Runtime knobs were frozen that way, while <see cref="KnobProfileScope"/> was
/// being pushed per request by the image and video services specifically to carry request settings into generation.
/// For every one of them that override silently did nothing — and worse than nothing, because a process-wide static
/// means whichever request touches the type first decides for every other request and both GPUs, which is exactly
/// the isolation the scope's <c>AsyncLocal</c> exists to provide.</para>
///
/// <para>There is deliberately no allowlist file. The knob's own declared scope is the allowlist: mark it
/// <see cref="KnobScope.Construction"/> and freezing is legal, because that scope says the value is baked in. That
/// keeps the declaration load-bearing instead of decorative, which is the whole point.</para></summary>
public sealed class KnobScopeIsEnforcedTests
{
    /// <summary>A knob cached into a field that can never be re-read.</summary>
    private static readonly Regex FrozenRead = new(
        @"static\s+readonly\s+[A-Za-z0-9_.<>?\[\]]+\s+[A-Za-z0-9_]+\s*=\s*EngineKnobs\.([A-Za-z0-9]+)\s*\.Value",
        RegexOptions.Compiled);

    /// <summary>Declared scope per knob, by the C# name a call site would write.</summary>
    private static Dictionary<string, KnobScope> DeclaredScopes()
    {
        Dictionary<string, KnobScope> scopes = new(StringComparer.Ordinal);
        foreach (FieldInfo field in typeof(EngineKnobs).GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            if (!field.FieldType.IsGenericType || field.FieldType.GetGenericTypeDefinition() != typeof(Knob<>))
            {
                continue;
            }
            object knob = field.GetValue(null)!;
            scopes[field.Name] = (KnobScope)knob.GetType().GetProperty(nameof(Knob<int>.Scope))!.GetValue(knob)!;
        }
        return scopes;
    }

    [Fact]
    public void A_Runtime_Knob_Is_Never_Frozen_Into_A_Static()
    {
        Dictionary<string, KnobScope> scopes = DeclaredScopes();
        Assert.NotEmpty(scopes);

        List<string> violations = [];
        string src = Path.Combine(RepoRoot.Path, "src");
        foreach (string file in Directory.EnumerateFiles(src, "*.cs", SearchOption.AllDirectories))
        {
            string relative = Path.GetRelativePath(RepoRoot.Path, file).Replace('\\', '/');
            if (relative.Contains("/obj/", StringComparison.Ordinal) || relative.Contains("/bin/", StringComparison.Ordinal))
            {
                continue;
            }
            foreach (Match match in FrozenRead.Matches(File.ReadAllText(file)))
            {
                string knob = match.Groups[1].Value;
                if (!scopes.TryGetValue(knob, out KnobScope scope))
                {
                    violations.Add($"{relative}: EngineKnobs.{knob} is not a declared knob");
                }
                else if (scope == KnobScope.Runtime)
                {
                    violations.Add($"{relative}: {knob} is KnobScope.Runtime");
                }
            }
        }

        Assert.True(violations.Count == 0,
            "A Runtime knob is frozen into a static readonly field, so a per-request override cannot reach it:\n  "
            + string.Join("\n  ", violations)
            + "\nEither read it live (`static X => EngineKnobs.K.Value;`) or, if the value really is baked in at "
            + "construction, declare the knob KnobScope.Construction and say why.");
    }

    /// <summary>Names the knobs that actually exercise the exemption, so redeclaring one Construction merely to
    /// silence the check above cannot pass unnoticed — it shows up here as a diff with a name on it.</summary>
    /// <remarks>Not a count of Construction knobs; plenty are declared Construction and never frozen, which is fine.
    /// This is the intersection that matters: declared bakeable AND actually baked.</remarks>
    [Fact]
    public void Only_These_Knobs_Are_Actually_Frozen()
    {
        Dictionary<string, KnobScope> scopes = DeclaredScopes();
        SortedSet<string> frozen = new(StringComparer.Ordinal);
        string src = Path.Combine(RepoRoot.Path, "src");
        foreach (string file in Directory.EnumerateFiles(src, "*.cs", SearchOption.AllDirectories))
        {
            string relative = Path.GetRelativePath(RepoRoot.Path, file).Replace('\\', '/');
            if (relative.Contains("/obj/", StringComparison.Ordinal) || relative.Contains("/bin/", StringComparison.Ordinal))
            {
                continue;
            }
            foreach (Match match in FrozenRead.Matches(File.ReadAllText(file)))
            {
                if (scopes.TryGetValue(match.Groups[1].Value, out KnobScope scope) && scope == KnobScope.Construction)
                {
                    frozen.Add(match.Groups[1].Value);
                }
            }
        }

        // All three are Vulkan device/instance decisions taken when the backend is built: the coopmat feature set is
        // baked into the pipelines compiled at construction, and the profiling and submit-per-op switches change how
        // the command stream is recorded from the first submit.
        Assert.Equal(["VkDisableCoopmat", "VkProfile", "VkSubmitPerOp"], frozen);
    }
}
