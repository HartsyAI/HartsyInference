using System.Text.RegularExpressions;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Source rules for writing a Vulkan op, checked without a GPU.
///
/// <para>The dispatch path enforces both of these at runtime, and that is where they bite hardest — but a runtime
/// guard only fires on a path some test actually runs, and Vulkan implements a fraction of the ops CUDA does, so
/// the ones most likely to be written next are the ones least likely to be covered. These run on the source.</para></summary>
public sealed class VulkanOpAuthoringLintTests
{
    private static string BackendSource() =>
        File.ReadAllText(Path.Combine(RepoRoot.Path, "src", "HartsyInference.Vulkan", "VulkanBackend.cs"));

    /// <summary>An interface default cannot be reached with <c>((IBackend)this).X(...)</c> from a class that
    /// declares <c>X</c> — the call binds back to the class and recurses until the stack ends.</summary>
    /// <remarks>Not hypothetical: three dtype fallbacks were written this way, and TROUBLESHOOTING records the same
    /// failure for <c>AffineBroadcastLastDim</c>. A fallback goes through the static reference implementation or
    /// the CPU backend, never through the interface.</remarks>
    [Fact]
    public void No_Override_Calls_Its_Own_Interface_Default()
    {
        string[] offenders = [.. BackendSource()
            .Split('\n')
            .Select((line, index) => (line, index))
            .Where(entry => entry.line.Contains("((IBackend)this)", StringComparison.Ordinal))
            .Select(entry => $"line {entry.index + 1}: {entry.line.Trim()}")];

        Assert.True(offenders.Length == 0,
            "An override cannot reach its own interface default; the call recurses to a stack overflow:\n  "
            + string.Join("\n  ", offenders));
    }

    /// <summary>Every public op that dispatches opens an op scope.</summary>
    /// <remarks>The scope is what suppresses the batched auto-flush. Without one, a flush can land between two
    /// dispatches of the same op and free the transients the later dispatches still read — the per-slice loops are
    /// exactly the shape that breaks. Twenty-seven ops were missing it when this rule was first enforced.</remarks>
    [Fact]
    public void Every_Dispatching_Op_Opens_A_Scope()
    {
        string source = BackendSource();
        // Split on member boundaries: a line at four-space indent starting an accessibility keyword.
        string[] members = Regex.Split(source, @"(?=\n    (?:public|internal) )");
        List<string> offenders = [];
        foreach (string member in members)
        {
            Match signature = Regex.Match(member, @"^\n    (?:public|internal) (?:unsafe )?(?:override )?[\w<>?\[\],. ]+ (\w+)\(");
            if (!signature.Success)
            {
                continue;
            }
            bool dispatches = Regex.IsMatch(member, @"\bDispatch\w*\(");
            if (dispatches && !member.Contains("EnterOp()", StringComparison.Ordinal))
            {
                offenders.Add(signature.Groups[1].Value);
            }
        }

        Assert.True(offenders.Count == 0,
            "These dispatch without opening an op scope (`using OpScope _op = EnterOp();` as the first "
            + "statement):\n  " + string.Join("\n  ", offenders));
    }
}
