using System.Reflection;
using HartsyInference.Core.Configuration;
using HartsyInference.Tests.Common;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>Makes <see cref="KnobScope"/> mean something. A <see cref="KnobScope.Runtime"/> knob is declared to be
/// "read each generation; safe to override per request" — so capturing one into storage that is written once and
/// never re-read freezes it at type-initialization and breaks its own declaration.
///
/// <para>This was not hypothetical. 54 Runtime knobs were frozen that way while <see cref="KnobProfileScope"/> was
/// being pushed per request by the image and video services specifically to carry request settings into generation.
/// For every one of them that override silently did nothing — and worse than nothing, because a process-wide static
/// means whichever request touches the type first decides for every other request and both GPUs, which is exactly
/// the isolation the scope's <c>AsyncLocal</c> exists to provide.</para>
///
/// <para><b>This parses C# rather than matching text, and that is load-bearing.</b> The first version of this test
/// was a regex requiring <c>EngineKnobs</c> immediately after the <c>=</c>, and it missed a live instance three
/// lines from one it caught: <c>static readonly bool AutoPromoteWeights = !EngineKnobs.NoAutopromote.Value;</c>,
/// where a single <c>!</c> was enough to hide a frozen Runtime knob. A syntax tree has no such blind spots, so the
/// negated initializer, the multi-line one, the <see cref="Lazy{T}"/> wrapper and the static constructor are all the
/// same shape to it: a knob read reached from storage that is written once.</para>
///
/// <para>There is deliberately no allowlist file. The knob's own declared scope is the allowlist: mark it
/// <see cref="KnobScope.Construction"/> and freezing is legal, because that scope says the value is baked in. That
/// keeps the declaration load-bearing instead of decorative, which is the whole point.</para></summary>
public sealed class KnobScopeIsEnforcedTests
{
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

    /// <summary>Every <c>EngineKnobs.Something.Value</c> read in a file, with the knob's name.</summary>
    private static IEnumerable<(MemberAccessExpressionSyntax Read, string Knob)> KnobReads(SyntaxNode root)
    {
        foreach (MemberAccessExpressionSyntax access in root.DescendantNodes().OfType<MemberAccessExpressionSyntax>())
        {
            if (access.Name.Identifier.ValueText != nameof(Knob<int>.Value)
                || access.Expression is not MemberAccessExpressionSyntax knob
                || knob.Expression is not IdentifierNameSyntax owner
                || owner.Identifier.ValueText != nameof(EngineKnobs))
            {
                continue;
            }
            yield return (access, knob.Name.Identifier.ValueText);
        }
    }

    /// <summary>Why this read is frozen, or null when it is evaluated afresh each time.</summary>
    /// <remarks>All three shapes are "written once, read forever". A property getter, a method body or a local is
    /// re-evaluated per call and is therefore fine — which is what the fix turned every violation into.</remarks>
    private static string? FreezeReason(SyntaxNode read)
    {
        foreach (SyntaxNode node in read.Ancestors())
        {
            switch (node)
            {
                case FieldDeclarationSyntax field
                    when field.Modifiers.Any(SyntaxKind.StaticKeyword) && field.Modifiers.Any(SyntaxKind.ReadOnlyKeyword):
                    return "a static readonly field initializer";
                case ConstructorDeclarationSyntax ctor when ctor.Modifiers.Any(SyntaxKind.StaticKeyword):
                    return "a static constructor";
                case ObjectCreationExpressionSyntax creation
                    when creation.Type.ToString().StartsWith("Lazy<", StringComparison.Ordinal):
                    return "a Lazy<T> initializer";
                // A getter, a method or a local re-reads on every access, so stop looking once inside one.
                case AccessorDeclarationSyntax:
                case MethodDeclarationSyntax:
                case ArrowExpressionClauseSyntax when node.Parent is PropertyDeclarationSyntax or MethodDeclarationSyntax:
                    return null;
            }
        }
        return null;
    }

    private static IEnumerable<string> SourceFiles(string directory)
    {
        foreach (string file in Directory.EnumerateFiles(Path.Combine(RepoRoot.Path, directory), "*.cs",
            SearchOption.AllDirectories))
        {
            string relative = Path.GetRelativePath(RepoRoot.Path, file).Replace('\\', '/');
            if (!relative.Contains("/obj/", StringComparison.Ordinal) && !relative.Contains("/bin/", StringComparison.Ordinal))
            {
                yield return file;
            }
        }
    }

    /// <summary>Every frozen knob read under <c>src/</c>, as (knob, where).</summary>
    private static List<(string Knob, string Where)> FrozenReads()
    {
        List<(string, string)> frozen = [];
        foreach (string file in SourceFiles("src"))
        {
            SyntaxNode root = CSharpSyntaxTree.ParseText(File.ReadAllText(file)).GetRoot();
            foreach ((MemberAccessExpressionSyntax read, string knob) in KnobReads(root))
            {
                if (FreezeReason(read) is string reason)
                {
                    string relative = Path.GetRelativePath(RepoRoot.Path, file).Replace('\\', '/');
                    frozen.Add((knob, $"{relative} ({reason})"));
                }
            }
        }
        return frozen;
    }

    [Fact]
    public void A_Runtime_Knob_Is_Never_Frozen()
    {
        Dictionary<string, KnobScope> scopes = DeclaredScopes();
        Assert.NotEmpty(scopes);

        List<string> violations = [.. FrozenReads()
            .Where(entry => !scopes.TryGetValue(entry.Knob, out KnobScope scope) || scope == KnobScope.Runtime)
            .Select(entry => $"{entry.Where}: {entry.Knob}")
            .Distinct()
            .Order(StringComparer.Ordinal)];

        Assert.True(violations.Count == 0,
            "A Runtime knob is captured into storage that is written once, so a per-request override cannot reach "
            + "it:\n  " + string.Join("\n  ", violations)
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
        string[] frozen = [.. FrozenReads()
            .Select(entry => entry.Knob)
            .Where(knob => scopes.TryGetValue(knob, out KnobScope scope) && scope == KnobScope.Construction)
            .Distinct()
            .Order(StringComparer.Ordinal)];

        // All three are Vulkan device/instance decisions taken when the backend is built: the coopmat feature set is
        // baked into the pipelines compiled at construction, and the profiling and submit-per-op switches change how
        // the command stream is recorded from the first submit.
        Assert.Equal(["VkDisableCoopmat", "VkProfile", "VkSubmitPerOp"], frozen);
    }
}
