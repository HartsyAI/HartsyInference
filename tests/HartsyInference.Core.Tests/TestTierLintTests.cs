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

    // A test is exempt from a lint if it demonstrably skips/guards when the resource is unavailable. Widen these sets
    // (or drop the magic comment `// tier-lint: guarded` on the offending line) rather than suppressing a real find.
    // The GPU set names only checks that ask whether a device is there. File and directory probes used to count
    // too, so the PtxDir() helper's Directory.Exists — which only finds compiled kernels — passed an unguarded
    // `new CudaBackend` file, and those tests failed on a machine without a GPU instead of being flagged here.
    private static readonly Regex RecognizedGpuGuard = new(
        @"CudaContext\.IsAvailable|BackendGate\.(TryOpen|UnavailableReason)|CudaAvailable|GpuAvailable|VulkanAvailable|"
        + @"HasCuda|HasGpu|TryCreate|SkipIfNo|tier-lint:\s*guarded",
        RegexOptions.Compiled);

    private static readonly Regex RecognizedFixtureGuard = new(
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
            GpuInit.IsMatch(body) && !RecognizedGpuGuard.IsMatch(body));

        Assert.True(violations.Count == 0,
            "These untagged (Unit-tier) tests instantiate a real GPU backend with no skip-guard, so they will hard-fail "
            + "on the GPU-less hosted CI runner. Add [Trait(\"Category\", \"GpuIntegration\")] (or a CudaContext.IsAvailable() / "
            + "BackendGate.TryOpen guard that returns early when no device is present):\n  " + string.Join("\n  ", violations));
    }

    [Fact]
    public void UntaggedUnitTests_DoNotReadGitignoredReferenceData_WithoutGuard()
    {
        List<string> violations = ScanUntaggedUnitFiles((path, body) =>
            RefDataPath.IsMatch(body) && FileReadApi.IsMatch(body) && !RecognizedFixtureGuard.IsMatch(body));

        Assert.True(violations.Count == 0,
            "These untagged (Unit-tier) tests read from tests/python-reference/ (reference tensors are gitignored via the "
            + "global *.bin rule and are absent on a fresh clone), with no skip-guard — they throw FileNotFoundException in "
            + "the Unit gate. Add [Trait(\"Category\", \"Integration\")] and a `if (!File.Exists(...)) return;` guard, or "
            + "commit the fixtures under a gitignore exception:\n  " + string.Join("\n  ", violations));
    }

    // The Vulkan device pin only holds if every backend a test opens goes through VulkanTestDevice. A bare constructor, a
    // target-typed new(), or a one-argument VulkanDevice.Create takes the default picker, which on the dev box is the RTX
    // 4090 (device 0), so a run pinned to the 3060 would still touch the other card.
    private static readonly Regex TargetTypedVulkanBackend = new(
        @"\bVulkanBackend\b\s*\??\s+[A-Za-z_]\w*\s*=\s*new\s*\(\s*\)", RegexOptions.Compiled);
    private static readonly Regex NewVulkanBackend = new(@"\bnew\s+VulkanBackend\s*\(\s*\)", RegexOptions.Compiled);
    private static readonly Regex CreateVulkanNull = new(@"\bCreateVulkan\s*\(\s*null\s*\)", RegexOptions.Compiled);
    private static readonly Regex VulkanDeviceCreate = new(@"\bVulkanDevice\.Create\s*\(", RegexOptions.Compiled);

    [Fact]
    public void Tests_OpenEveryVulkanBackendThroughTheDeviceHelper()
    {
        List<string> violations = new();
        string testsRoot = Path.Combine(RepoRoot.Path, "tests");
        foreach (string file in Directory.EnumerateFiles(testsRoot, "*.cs", SearchOption.AllDirectories))
        {
            if (Path.GetFileName(file) == "VulkanTestDevice.cs"
                || file.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}")
                || file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}"))
                continue;
            string body = BlankComments(File.ReadAllText(file));
            foreach (string use in DefaultPickerUses(body))
                violations.Add($"{Path.GetRelativePath(RepoRoot.Path, file)}: {use}");
        }
        violations.Sort(StringComparer.Ordinal);
        Assert.True(violations.Count == 0,
            "These test files open a Vulkan backend on the default picker, so HARTSY_TEST_VULKAN_DEVICE does not pin it "
            + "(on the dev box the default is the RTX 4090). Open it with VulkanTestDevice.Create() or "
            + "VulkanTestDevice.CreateDevice(instance) instead:\n  " + string.Join("\n  ", violations));
    }

    /// <summary>Default-picker constructions in comment-free source, as "line N: what". Matching runs over the whole file,
    /// so a target-typed new() split across lines is still found.</summary>
    private static IEnumerable<string> DefaultPickerUses(string body)
    {
        foreach (Match m in TargetTypedVulkanBackend.Matches(body))
            yield return $"line {LineOf(body, m.Index)}: target-typed new() for VulkanBackend";
        foreach (Match m in NewVulkanBackend.Matches(body))
            yield return $"line {LineOf(body, m.Index)}: VulkanBackend constructed with no ordinal";
        foreach (Match m in CreateVulkanNull.Matches(body))
            yield return $"line {LineOf(body, m.Index)}: CreateVulkan with a null ordinal";
        foreach (Match m in VulkanDeviceCreate.Matches(body))
        {
            if (ArgumentCount(body, m.Index + m.Length) == 1)
                yield return $"line {LineOf(body, m.Index)}: VulkanDevice.Create with one argument";
        }
    }

    /// <summary>Top-level argument count of the call whose opening parenthesis is just before <paramref name="start"/>.</summary>
    private static int ArgumentCount(string body, int start)
    {
        int depth = 1, commas = 0;
        for (int i = start; i < body.Length && depth > 0; i++)
        {
            char c = body[i];
            if (c == '(') depth++;
            else if (c == ')') depth--;
            else if (c == ',' && depth == 1) commas++;
        }
        return commas + 1;
    }

    private static int LineOf(string body, int index)
    {
        int line = 1;
        for (int i = 0; i < index; i++)
            if (body[i] == '\n') line++;
        return line;
    }

    /// <summary>Blanks comments in place and keeps every newline, so the line numbers reported match the file.</summary>
    private static string BlankComments(string source) =>
        Regex.Replace(source, @"/\*.*?\*/|//[^\n]*", m => Regex.Replace(m.Value, @"[^\n]", " "), RegexOptions.Singleline);

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
