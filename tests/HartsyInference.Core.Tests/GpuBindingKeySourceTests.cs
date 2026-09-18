using System.Text.RegularExpressions;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>Asserts the whole solution allocates GPU binding keys from one sequence.
///
/// <para>The invariant is not "the counter is thread-safe" — each copy was. It is that there is only ONE. A binding
/// key names a bucket in a tensor's binding table, and a tensor resident on two devices at once holds one binding per
/// cache; two counters that each begin at 1 hand both caches the same key, so one backend's teardown clears the
/// other's binding and the surviving backend reads freed memory. That has now been written twice — once as a static
/// inside a generic class, where it silently became one counter per closed type, and once as a private field in the
/// CUDA state registry. Neither was visible until a second backend existed, which is precisely when it is too late.</para></summary>
/// <remarks>A source scan rather than a runtime check: the collision only manifests with two GPU backends live on
/// one machine, so nothing that runs in CI would ever observe it.</remarks>
public sealed class GpuBindingKeySourceTests
{
    /// <summary>A counter whose name says it numbers keys. Narrow on purpose — <c>Interlocked.Increment</c> itself is
    /// ordinary and appears throughout the caches for hit and sync counters.</summary>
    private static readonly Regex KeyCounter = new(@"\b_next\w*Key\b", RegexOptions.Compiled);

    /// <summary>The one file allowed to hold the sequence.</summary>
    private const string Allowed = "src/HartsyInference.Core/Tensors/GpuBindingKeys.cs";

    [Fact]
    public void Only_One_File_Allocates_Gpu_Binding_Keys()
    {
        List<string> offenders = [];
        string src = Path.Combine(RepoRoot.Path, "src");
        foreach (string file in Directory.EnumerateFiles(src, "*.cs", SearchOption.AllDirectories))
        {
            string rel = Path.GetRelativePath(RepoRoot.Path, file).Replace('\\', '/');
            if (rel.Contains("/obj/", StringComparison.Ordinal) || rel.Contains("/bin/", StringComparison.Ordinal)
                || rel == Allowed)
            {
                continue;
            }
            if (KeyCounter.IsMatch(File.ReadAllText(file)))
            {
                offenders.Add(rel);
            }
        }

        Assert.True(offenders.Count == 0,
            $"These files declare their own GPU binding-key counter: {string.Join(", ", offenders)}. "
            + $"Call GpuBindingKeys.Next() instead — a second sequence gives two caches the same key.");
    }
}
