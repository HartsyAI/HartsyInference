using HartsyInference.Core.Backends;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>MovieRoll's streaming forwards on CUDA against the CPU run, in two GEMM lanes; the shared bodies are in <see cref="MovieRollStreamingScenarios"/>.
/// Runs as a GPU integration test (Category=GpuIntegration); it is skipped where no CUDA device is present.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class MovieRollStreamingGpuTests(ITestOutputHelper output)
{
    private const float ExactTolerance = 5e-4f;
    private const float Tf32Tolerance = 2e-3f;

    [CudaDeviceFact]
    public void Cuda_StreamEncode_ExactGemms_ReadsNothingBackAndMatchesTheCpuCache() =>
        new MovieRollStreamingScenarios(output).RunEncode(GpuDevices.OpenCuda, exactGemm: true, ExactTolerance);

    [CudaDeviceFact]
    public void Cuda_StreamEncode_Tf32Gemms_ReadsNothingBackAndMatchesTheCpuCache() =>
        new MovieRollStreamingScenarios(output).RunEncode(GpuDevices.OpenCuda, exactGemm: false, Tf32Tolerance);

    [CudaDeviceFact]
    public void Cuda_StreamDenoise_ExactGemms_ReadsNothingBackAndMatchesTheCpuRun() =>
        new MovieRollStreamingScenarios(output).RunDenoise(GpuDevices.OpenCuda, exactGemm: true, ExactTolerance);

    [CudaDeviceFact]
    public void Cuda_StreamDenoise_Tf32Gemms_ReadsNothingBackAndMatchesTheCpuRun() =>
        new MovieRollStreamingScenarios(output).RunDenoise(GpuDevices.OpenCuda, exactGemm: false, Tf32Tolerance);

    [CudaDeviceFact]
    public void Cuda_StreamRecomputeFiveChunks_ExactGemms_ReadsNothingBackAndMatchesTheCpuCache() =>
        new MovieRollStreamingScenarios(output).RunRecompute(GpuDevices.OpenCuda, exactGemm: true, [1, 1, 1, 1, 1], ExactTolerance);

    [CudaDeviceFact]
    public void Cuda_StreamRecomputeFiveChunks_Tf32Gemms_ReadsNothingBackAndMatchesTheCpuCache() =>
        new MovieRollStreamingScenarios(output).RunRecompute(GpuDevices.OpenCuda, exactGemm: false, [1, 1, 1, 1, 1], Tf32Tolerance);

    [CudaDeviceFact]
    public void Cuda_StreamRecomputeOneChunk_ExactGemms_ReadsNothingBackAndMatchesTheCpuCache() =>
        new MovieRollStreamingScenarios(output).RunRecompute(GpuDevices.OpenCuda, exactGemm: true, [5], ExactTolerance);

    [CudaDeviceFact]
    public void Cuda_StreamRecomputeOneChunk_Tf32Gemms_ReadsNothingBackAndMatchesTheCpuCache() =>
        new MovieRollStreamingScenarios(output).RunRecompute(GpuDevices.OpenCuda, exactGemm: false, [5], Tf32Tolerance);
}
