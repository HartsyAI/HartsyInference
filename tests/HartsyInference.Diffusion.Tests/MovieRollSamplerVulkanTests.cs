using HartsyInference.Core.Backends;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The whole MovieRoll sampler on Vulkan against the CPU run, on the device's default GEMM path; the bodies are in
/// <see cref="MovieRollSamplerScenarios"/>. Measured on a local RTX 3060 (Vulkan) on 2026-11-27: the largest latent error is 9.5e-7.
/// Runs as a GPU integration test (Category=GpuIntegration); it is skipped where no Vulkan device is present.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class MovieRollSamplerVulkanTests(ITestOutputHelper output)
{
    private const float Tolerance = 1e-5f;

    [VulkanDeviceFact]
    public void Vulkan_FullSampler_ReadsNothingBackAndMatchesTheCpuLatents() =>
        new MovieRollSamplerScenarios(output).Run(GpuDevices.OpenVulkan, exactGemm: null, Tolerance);
}
