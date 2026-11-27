using HartsyInference.Core.Backends;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>MovieRoll's streaming forwards on Vulkan against the CPU run, on the device's default GEMM path; the shared bodies are in <see cref="MovieRollStreamingScenarios"/>.
/// Runs as a GPU integration test (Category=GpuIntegration); it is skipped where no Vulkan device is present.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class MovieRollStreamingVulkanTests(ITestOutputHelper output)
{
    private const float ExactTolerance = 1e-5f;

    [VulkanDeviceFact]
    public void Vulkan_StreamEncode_ReadsNothingBackAndMatchesTheCpuCache() =>
        new MovieRollStreamingScenarios(output).RunEncode(GpuDevices.OpenVulkan, exactGemm: null, ExactTolerance);

    [VulkanDeviceFact]
    public void Vulkan_StreamDenoise_ReadsNothingBackAndMatchesTheCpuRun() =>
        new MovieRollStreamingScenarios(output).RunDenoise(GpuDevices.OpenVulkan, exactGemm: null, ExactTolerance);

    [VulkanDeviceFact]
    public void Vulkan_StreamRecomputeFiveChunks_ReadsNothingBackAndMatchesTheCpuCache() =>
        new MovieRollStreamingScenarios(output).RunRecompute(GpuDevices.OpenVulkan, exactGemm: null, [1, 1, 1, 1, 1], ExactTolerance);

    [VulkanDeviceFact]
    public void Vulkan_StreamRecomputeOneChunk_ReadsNothingBackAndMatchesTheCpuCache() =>
        new MovieRollStreamingScenarios(output).RunRecompute(GpuDevices.OpenVulkan, exactGemm: null, [5], ExactTolerance);
}
