using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Cuda;
using HartsyInference.LLM.Kimi;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>The synthetic Kimi decoder on CUDA and Vulkan against the CPU backend, on the same weights and tokens. Each test reports itself
/// as skipped when its device is absent.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class KimiK31ModelGpuTests
{
    private const float Tol = 2e-3f;
    private static readonly int[] Tokens = [1, 5, 7, 2, 11, 3];

    private readonly ITestOutputHelper _output;
    public KimiK31ModelGpuTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Cuda_LogitsMatchCpu()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDir());
        AssertMatchesCpu(backend);
    }

    [Fact]
    public void Vulkan_LogitsMatchCpu()
    {
        if (!VulkanContext.IsAvailable()) { _output.WriteLine("SKIPPED: Vulkan unavailable"); return; }
        using VulkanBackend backend = new();
        AssertMatchesCpu(backend);
    }

    private static void AssertMatchesCpu(IBackend device)
    {
        KimiK31Weights weights = KimiK31ModelTests.RandomWeights(new Random(9));
        float[] cpu = Run(new CpuBackend(), weights);
        float[] gpu = Run(device, weights);
        for (int i = 0; i < cpu.Length; i++)
            Assert.True(MathF.Abs(cpu[i] - gpu[i]) <= Tol * MathF.Max(1f, MathF.Abs(cpu[i])), $"logit[{i}] cpu {cpu[i]} device {gpu[i]}");
    }

    private static float[] Run(IBackend backend, KimiK31Weights weights)
    {
        using KimiK31Model model = new(backend, KimiK31ModelTests.Config, weights, maxPositions: 16);
        using KimiK31State state = new(KimiK31ModelTests.Config, capacity: 16);
        return model.Forward(Tokens, state);
    }

    private static string PtxDir()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        return Path.Combine(AppContext.BaseDirectory, "Ptx");
    }
}
