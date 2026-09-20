using System.Text.RegularExpressions;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Gpu.Tests;

/// <summary>A source rule for every backend on the shared base, checked without a GPU.</summary>
public sealed class OpScopeLintTests
{
    private static readonly string[] _backendSources =
    [
        Path.Combine("src", "HartsyInference.Cuda", "CudaBackend.cs"),
        Path.Combine("src", "HartsyInference.Vulkan", "VulkanBackend.cs"),
    ];

    /// <summary>The scope has to be held, not discarded.</summary>
    /// <remarks>Before the shared base, CUDA's op entry was a void call and every op invoked it as a bare
    /// statement. Under the base that spelling still compiles — <c>EnterOp()</c> returns a struct and C# is happy
    /// to throw it away — but the scope's dispose is what decrements the depth, so a discarded one raises the
    /// depth permanently. Every later op then believes it is nested: no finalizer drain, no orphan sweep, and on a
    /// backend that batches submits, no flush. Nothing fails at the call site, and the effect is unbounded.</remarks>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void An_Op_Scope_Is_Never_Discarded(int sourceIndex)
    {
        string path = Path.Combine(RepoRoot.Path, _backendSources[sourceIndex]);
        string[] lines = File.ReadAllLines(path);
        List<string> offenders = [];
        for (int index = 0; index < lines.Length; index++)
        {
            // A call whose result goes nowhere: `EnterOp();` as a whole statement, rather than `using ... =`.
            if (Regex.IsMatch(lines[index], @"^\s*(?:\w+\.)?EnterOp\(\s*\)\s*;\s*$"))
            {
                offenders.Add($"{Path.GetFileName(path)}:{index + 1}: {lines[index].Trim()}");
            }
        }

        Assert.True(offenders.Count == 0,
            "An op scope must be held for the length of the op (`using OpScope _op = EnterOp();`). Discarding it "
            + "raises the op depth forever, so every later op is treated as nested:\n  "
            + string.Join("\n  ", offenders));
    }
}
