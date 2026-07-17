using System.Text;
using System.Text.RegularExpressions;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests;

/// <summary>Meta-tests that enforce the test-tier contract from <c>docs/CODE_STYLE.md</c> so tier drift cannot
/// silently return. The hosted-CI Unit gate runs every test that carries no <c>[Trait("Category", ...)]</c>, on a
/// machine with no GPU and no checkpoints. A test that reaches for one of those resources must therefore either
/// carry a tier trait (so it is excluded from the gate and picked up by the GPU/nightly lane) or skip cleanly when
/// the resource is absent. These lints catch the two failure modes that have actually bitten us: an untagged test
/// that hard-instantiates a GPU backend (crashes the GPU-less runner), and an untagged test that reads gitignored
/// <c>python-reference</c> fixtures with no guard (<c>FileNotFoundException</c> on a fresh clone — the exact bug that
/// motivated this suite audit). The lints are deliberately narrow and comment-stripped to keep false positives at
/// zero; widen the recognized-guard set below rather than adding a name-substring blocklist.</summary>
public sealed class TestTierLintTests
{
    private readonly ITestOutputHelper _output;
    public TestTierLintTests(ITestOutputHelper output) => _output = output;

    private static readonly Regex TierTrait = new(
        """Trait\("Category", "(Integration|GpuIntegration|Slow|SyntheticSmoke)"\)|Trait\("Network", "Real"\)""",
        RegexOptions.Compiled);
    private static readonly Regex HasFacts = new(@"\[Fact\]|\[Theory\]", RegexOptions.Compiled);

    // A test is exempt from a lint if it demonstrably skips/guards when the resource is unavailable. Widen this set
    // (or drop the magic comment `// tier-lint: guarded` on the offending line) rather than suppressing a real find.
    private static readonly Regex RecognizedGuard = new(
        @"CudaAvailable|GpuAvailable|VulkanAvailable|HasCuda|HasGpu|TryCreate|SkipIfNo|IsAvailable|FixturesPresent|"
        + @"catch\s*\(|Directory\.Exists|File\.Exists|GetEnvironmentVariable|tier-lint:\s*guarded",
        RegexOptions.Compiled);

    private static readonly Regex GpuInit = new(@"new\s+CudaBackend|new\s+VulkanBackend", RegexOptions.Compiled);
    private static readonly Regex RefDataPath = new(@"python-reference", RegexOptions.Compiled);
    private static readonly Regex FileReadApi = new(@"ReadAllBytes|File\.OpenRead|new\s+BinaryReader", RegexOptions.Compiled);

    [Fact]
    public void UntaggedUnitTests_DoNotInstantiateGpuBackend_WithoutGuard()
    {
        List<string> violations = ScanUntaggedUnitFiles((path, body) =>
            GpuInit.IsMatch(body) && !RecognizedGuard.IsMatch(body));

        Assert.True(violations.Count == 0,
            "These untagged (Unit-tier) tests instantiate a real GPU backend with no skip-guard, so they will hard-fail "
            + "on the GPU-less hosted CI runner. Add [Trait(\"Category\", \"GpuIntegration\")] (or a CudaAvailable()-style "
            + "guard that returns early when no device is present):\n  " + string.Join("\n  ", violations));
    }

    [Fact]
    public void UntaggedUnitTests_DoNotReadGitignoredReferenceData_WithoutGuard()
    {
        List<string> violations = ScanUntaggedUnitFiles((path, body) =>
            RefDataPath.IsMatch(body) && FileReadApi.IsMatch(body) && !RecognizedGuard.IsMatch(body));

        Assert.True(violations.Count == 0,
            "These untagged (Unit-tier) tests read from tests/python-reference/ (reference tensors are gitignored via the "
            + "global *.bin rule and are absent on a fresh clone), with no skip-guard — they throw FileNotFoundException in "
            + "the Unit gate. Add [Trait(\"Category\", \"Integration\")] and a `if (!File.Exists(...)) return;` guard, or "
            + "commit the fixtures under a gitignore exception:\n  " + string.Join("\n  ", violations));
    }

    private List<string> ScanUntaggedUnitFiles(Func<string, string, bool> isViolation)
    {
        string testsRoot = Path.Combine(RepoRoot.Path, "tests");
        List<string> violations = new();
        foreach (string file in Directory.EnumerateFiles(testsRoot, "*.cs", SearchOption.AllDirectories))
        {
            if (file.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}")
                || file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}"))
                continue;
            string raw = File.ReadAllText(file);
            if (!HasFacts.IsMatch(raw) || TierTrait.IsMatch(raw))
                continue;
            string body = StripComments(raw);
            if (isViolation(file, body))
                violations.Add(Path.GetRelativePath(RepoRoot.Path, file));
        }
        violations.Sort(StringComparer.Ordinal);
        return violations;
    }

    /// <summary>Removes block and line comments so a resource mentioned only in prose (a TODO pointing at a future
    /// fixture, a "diffed against neucodec_ref.py" note) is not mistaken for a runtime dependency.</summary>
    private static string StripComments(string source)
    {
        string noBlocks = Regex.Replace(source, @"/\*.*?\*/", string.Empty, RegexOptions.Singleline);
        return Regex.Replace(noBlocks, @"//[^\n]*", string.Empty);
    }
}
