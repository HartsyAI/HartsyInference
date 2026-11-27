using HartsyInference.Core.Backends;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The whole MovieRoll sampler on CUDA against the CPU run, in two GEMM lanes. Measured on a 4090 (#1295, 10/10, zero device reads): exact 3.196e-4
/// against the 5e-4 bound, TF32 1.381e-3 against the 2e-3 bound; the bodies are in <see cref="MovieRollSamplerScenarios"/>.
/// Runs as a GPU integration test (Category=GpuIntegration); it is skipped where no CUDA device is present.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class MovieRollSamplerGpuTests(ITestOutputHelper output)
{
    /// <summary>Exact-lane bound, the same as the streaming facts. The full sampler measured 3.196e-4 on a 4090 (#1295, 10/10 passing, zero device reads):
    /// a 36% margin. The streaming facts' measured maximum is 3.7e-4 (recompute, one chunk, block 2 keys), at 800f8a182 and 5a22be77c.</summary>
    private const float ExactTolerance = 5e-4f;

    /// <summary>TF32-lane bound, the same as the streaming facts. The full sampler measured 1.381e-3 on the same 4090 run: a 31% margin. The streaming
    /// facts' measured maximum is 1.479e-3 (recompute, one chunk, block 2 keys), at 800f8a182 and 5a22be77c.</summary>
    private const float Tf32Tolerance = 2e-3f;

    [CudaDeviceFact]
    public void Cuda_FullSampler_ExactGemms_ReadsNothingBackAndMatchesTheCpuLatents() =>
        new MovieRollSamplerScenarios(output).Run(GpuDevices.OpenCuda, exactGemm: true, ExactTolerance);

    [CudaDeviceFact]
    public void Cuda_FullSampler_Tf32Gemms_ReadsNothingBackAndMatchesTheCpuLatents() =>
        new MovieRollSamplerScenarios(output).Run(GpuDevices.OpenCuda, exactGemm: false, Tf32Tolerance);
}
