using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>The paged cache on Vulkan, where every attention call reads its contiguous prefix: the checks
/// <see cref="PagedKvCacheCudaTests"/> runs on CUDA, on the same data.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class PagedKvCacheVulkanTests
{
    [VulkanFact]
    public void ThePrefixMatchesAContiguousCache()
    {
        using VulkanBackend backend = new();
        PagedKvCacheCudaTests.PrefixMatchesAContiguousCache(backend, "none", 0);
    }

    [VulkanFact]
    public void ThePrefixSizedAheadMatchesAContiguousCache()
    {
        using VulkanBackend backend = new();
        PagedKvCacheCudaTests.PrefixMatchesAContiguousCache(backend, "none", 101);
    }

    [VulkanFact]
    public void ThePagesOfTwoSequencesHoldTheirOwnRows()
    {
        using VulkanBackend backend = new();
        PagedKvCacheCudaTests.PagesOfTwoSequencesHoldTheirOwnRows(backend, "none");
    }
}
