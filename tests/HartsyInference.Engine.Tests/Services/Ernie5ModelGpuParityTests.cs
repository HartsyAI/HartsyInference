using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Cuda;
using HartsyInference.LLM.Ernie5;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The whole ERNIE-5.5 host model on a GPU backend against the CPU: the synthetic checkpoint's logits after a prefill chunk,
/// through every decoder layer's device attention and MoE, must match the CPU run. Each test is reported as skipped (never passed)
/// when its device or kernels are absent.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class Ernie5ModelGpuParityTests : IDisposable
{
    private static readonly int[] Tokens = [1, 2, 3, 4, 5];

    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-ernie5-gpu-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [ErnieCudaFact]
    public void Cuda_SyntheticErnie_LogitsMatchCpu() => ParityOn(() => new CudaBackend(0, PtxDirectory()), "CUDA", resident: false);

    [ErnieCudaFact]
    public void Cuda_SyntheticErnie_ResidentWeights_LogitsMatchCpu() => ParityOn(() => new CudaBackend(0, PtxDirectory()), "CUDA", resident: true);

    [ErnieVulkanFact]
    public void Vulkan_SyntheticErnie_LogitsMatchCpu() => ParityOn(() => new VulkanBackend(), "Vulkan", resident: false);

    // Prefill then one decode step on the same state, so the HCA and DSA caches are read across steps as well as written.
    private void ParityOn(Func<IBackend> open, string device, bool resident)
    {
        string dir = WriteCheckpoint();
        using CpuBackend cpu = new();
        float[] want = Logits(dir, cpu, resident);
        using IBackend gpu = open();
        AssertClose(want, Logits(dir, gpu, resident), device);
    }

    private string WriteCheckpoint()
    {
        string dir = Path.Combine(_root, Guid.NewGuid().ToString("N"));
        Ernie5SyntheticCheckpoint.Write(dir);
        return dir;
    }

    // A prefill chunk with every row's logits, then one decode step on the same state; both are compared, not just the last token.
    private static float[] Logits(string dir, IBackend backend, bool resident)
    {
        using Ernie5Model model = Ernie5Model.Load(dir, backend, new Ernie5LoadOptions
        {
            MaxSequenceTokens = Tokens.Length + 1, ResidentWeights = resident,
        });
        using Ernie5SequenceState state = model.CreateState(Tokens.Length + 1);
        float[] prefill = model.Forward(Tokens, state, allRows: true);
        float[] decode = model.Forward([Tokens[^1] + 1], state);
        return [.. prefill, .. decode];
    }

    // Tolerance for F32 weights, where the device and the CPU differ only in accumulation order. FP8 weights will need a looser bound.
    private static void AssertClose(float[] want, float[] got, string device)
    {
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
        {
            Assert.True(float.IsFinite(got[i]), $"{device} logit {i} is {got[i]}");
            float tol = 1e-3f * MathF.Max(1f, MathF.Abs(want[i]));
            Assert.True(MathF.Abs(want[i] - got[i]) <= tol, $"{device} logit {i}: cpu {want[i]}, device {got[i]}");
        }
    }

    // The PTX beside the test assembly, else the repository's Cuda Ptx folder.
    private static string PtxDirectory()
    {
        string beside = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (Directory.Exists(beside)) return beside;
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        throw new DirectoryNotFoundException("No CUDA PTX directory beside the test assembly or in the repository.");
    }

    // Probed once per test run; each attribute reads the cached result instead of opening a device during discovery.
    private static readonly Lazy<string?> CudaFailure = new(CudaProbe);
    private static readonly Lazy<string?> VulkanFailure = new(VulkanProbe);

    private static string? CudaProbe()
    {
        try
        {
            string ptx = PtxDirectory();
            if (!File.Exists(Path.Combine(ptx, "hadamard.ptx")) || !File.Exists(Path.Combine(ptx, "fp8_block_linear.ptx")))
                return $"missing in {ptx}";
            using CudaBackend backend = new(0, ptx);
            return null;
        }
        catch (Exception ex)
        {
            return $"{ex.GetType().Name}: {ex.Message}";
        }
    }

    private static string? VulkanProbe()
    {
        try
        {
            string spv = Path.Combine(AppContext.BaseDirectory, "Spirv");
            if (!File.Exists(Path.Combine(spv, "hadamard.spv")) || !File.Exists(Path.Combine(spv, "fp8_block_linear.spv")))
                return "hadamard.spv / fp8_block_linear.spv not built";
            using VulkanBackend backend = new();
            return null;
        }
        catch (Exception ex)
        {
            return $"{ex.GetType().Name}: {ex.Message}";
        }
    }

    /// <summary>A fact reported as skipped (never passed) when no CUDA device opens or the block-op PTX is missing.</summary>
    private sealed class ErnieCudaFactAttribute : FactAttribute
    {
        public ErnieCudaFactAttribute()
        {
            string? failure = CudaFailure.Value;
            if (failure is not null) Skip = $"SKIPPED: no usable CUDA device or block-op PTX ({failure}); GPU check did not run.";
        }
    }

    /// <summary>A fact reported as skipped (never passed) when no Vulkan device opens or the block-op SPIR-V is not built.</summary>
    private sealed class ErnieVulkanFactAttribute : FactAttribute
    {
        public ErnieVulkanFactAttribute()
        {
            string? failure = VulkanFailure.Value;
            if (failure is not null) Skip = $"SKIPPED: no usable Vulkan device or block-op SPIR-V ({failure}); GPU check did not run.";
        }
    }
}
