using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;
using Checks = HartsyInference.Core.Tests.Backends.BlockSparseAttentionDeviceChecks;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Block-sparse attention on Vulkan against the CPU reference and against LongCat-Video's kernel
/// (<see cref="BlockSparseAttentionDeviceChecks"/>).</summary>
[Trait("Category", "GpuIntegration")]
public sealed class BlockSparseAttentionVulkanTests(ITestOutputHelper log)
{
    [VulkanTheory]
    [MemberData(nameof(Checks.ReferenceCases), MemberType = typeof(Checks))]
    public void MatchesTheReferenceKernelsVectors(string name)
    {
        using VulkanBackend vulkan = new();
        Checks.MatchesTheReferenceKernelsVectors(vulkan, log, name);
    }

    [VulkanTheory]
    [MemberData(nameof(Checks.ReferenceCases), MemberType = typeof(Checks))]
    public void MatchesTheCpuReferenceInEveryDtype(string name)
    {
        using VulkanBackend vulkan = new();
        Checks.MatchesTheCpuReferenceInEveryDtype(vulkan, log, name);
    }

    [VulkanTheory]
    [MemberData(nameof(Checks.RandomShapes), MemberType = typeof(Checks))]
    public void RandomShapesMatchTheCpuReference(int seed, int batch, int heads, int dim, int[] queryGrid, int[] keyGrid,
        int[] queryBlock, int[] keyBlock, double sparsity, float scale)
    {
        using VulkanBackend vulkan = new();
        Checks.MatchesTheCpuReferenceOnRandomShapes(vulkan, log, seed, batch, heads, dim, queryGrid, keyGrid, queryBlock, keyBlock,
            sparsity, scale);
    }

    [VulkanFact]
    public void SelectsAsTheReferenceDoesAmongManyKeyBlocks()
    {
        using VulkanBackend vulkan = new();
        Checks.SelectsAsTheReferenceDoesAmongManyKeyBlocks(vulkan, log);
    }

    [VulkanTheory]
    [MemberData(nameof(Checks.SpecialCases), MemberType = typeof(Checks))]
    public void SpecialScoresMatchTheCpuReference(string name)
    {
        using VulkanBackend vulkan = new();
        Checks.SpecialScoresMatchTheCpuReference(vulkan, log, name);
    }

    [VulkanFact]
    public void ResidentOperandsAreReadAndNothingIsSyncedUntilAResultIs()
    {
        using VulkanBackend vulkan = new();
        Checks.ResidentOperandsAreReadAndNothingIsSyncedUntilAResultIs(vulkan);
    }

    [VulkanFact]
    public void TheMostKeyBlocksTheSelectionHoldsStillRun()
    {
        using VulkanBackend vulkan = new();
        Checks.TheMostKeyBlocksStillRun(vulkan, log, 7680);
    }

    [VulkanFact]
    public void ACallSplitIntoManySubmissionsMatchesTheCpuReference()
    {
        // One unit of work a submission: every workgroup along x of the selection and the attention goes alone.
        using VulkanBackend vulkan = new() { BlockSparseWorkPerSubmit = 1 };
        Checks.MatchesTheCpuReferenceOnRandomShapes(vulkan, log, seed: 9, batch: 2, heads: 3, dim: 32, queryGrid: [4, 8, 16],
            keyGrid: [8, 8, 8], queryBlock: [4, 4, 8], keyBlock: [4, 4, 4], sparsity: 0.75, scale: 0f);
    }

    [VulkanFact]
    public void NearTiedBlocksAreSelectedAsTheReferenceDoes()
    {
        using VulkanBackend vulkan = new();
        Checks.NearTiedBlocksAreSelectedAsTheReferenceDoes(vulkan);
    }

    [VulkanFact]
    public void TooWideAHeadAndTooManyKeyBlocksAreRefusedBeforeAnyWrite()
    {
        using VulkanBackend vulkan = new();
        Checks.Refused(vulkan, new() { QueryGrid = new(4, 4, 4), KeyGrid = new(4, 4, 4), Sparsity = 0.0 }, 129, "got 129 and 1.");
        Checks.Refused(vulkan, new() { QueryGrid = new(4, 4, 4), KeyGrid = new(4, 4, 4 * 7681), Sparsity = 0.9 }, 16, "got 16 and 7681.");
    }

    private sealed class VulkanFactAttribute : FactAttribute
    {
        public VulkanFactAttribute() => Skip = Checks.VulkanSkip;
    }

    private sealed class VulkanTheoryAttribute : TheoryAttribute
    {
        public VulkanTheoryAttribute() => Skip = Checks.VulkanSkip;
    }
}
