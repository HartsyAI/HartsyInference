using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Ernie5;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The whole ERNIE-5.5 host model on a GPU backend against the CPU: the synthetic checkpoint's logits after a prefill chunk and a decode step,
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

    [ErnieVulkanFact]
    public void Vulkan_SyntheticErnie_ResidentWeights_LogitsMatchCpu() => ParityOn(() => new VulkanBackend(), "Vulkan", resident: true);

    // The FP8 checkpoint keeps its linears packed: each runs IBackend.Fp8BlockLinear on the device, against the CPU running the same file.
    [ErnieCudaFact]
    public void Cuda_SyntheticErnieFp8_LogitsMatchCpu() =>
        ParityOn(() => new CudaBackend(0, PtxDirectory()), "CUDA", resident: false, Ernie5SyntheticCheckpoint.Storage.Fp8);

    [ErnieCudaFact]
    public void Cuda_SyntheticErnieFp8_ResidentWeights_LogitsMatchCpu() =>
        ParityOn(() => new CudaBackend(0, PtxDirectory()), "CUDA", resident: true, Ernie5SyntheticCheckpoint.Storage.Fp8);

    [ErnieVulkanFact]
    public void Vulkan_SyntheticErnieFp8_ResidentWeights_LogitsMatchCpu() =>
        ParityOn(() => new VulkanBackend(), "Vulkan", resident: true, Ernie5SyntheticCheckpoint.Storage.Fp8);

    [ErnieVulkanFact]
    public void Vulkan_SyntheticErnieFp8_LogitsMatchCpu() =>
        ParityOn(() => new VulkanBackend(), "Vulkan", resident: false, Ernie5SyntheticCheckpoint.Storage.Fp8);

    // The real checkpoint's dimensions, FP8, three layers (HCA, DSA, HCA) and two MoE layers: the shapes and kernel limits of the rental run.
    // High-precision GEMM so the F32 projections (router, LM head, head mixes) run without TF32; otherwise their ~1e-3 rounding, not the FP8
    // path, is what this comparison measures.
    [ErnieCudaFact]
    public void Cuda_RealDimensionsFp8_LogitsMatchCpu() => ParityOn(() => new CudaBackend(0, PtxDirectory()) { HighPrecisionGemm = true }, "CUDA",
        resident: false, Ernie5SyntheticCheckpoint.Storage.Fp8, Ernie5SyntheticCheckpoint.RealDimensionOverrides);

    // F16 GEMM off, the Vulkan counterpart of HighPrecisionGemm above: it rounds GEMM inputs to F16, which is the ~1e-3 this compares.
    [ErnieVulkanFact]
    public void Vulkan_RealDimensionsFp8_LogitsMatchCpu() => ParityOn(() => new VulkanBackend { EnableF16Gemm = false }, "Vulkan", resident: false,
        Ernie5SyntheticCheckpoint.Storage.Fp8, Ernie5SyntheticCheckpoint.RealDimensionOverrides);

    // One FP8 linear whose weight spans two row blocks and three column blocks with distinct scales, on the device and the CPU.
    [ErnieCudaFact]
    public void Cuda_Fp8LinearWithMultipleBlocks_MatchesCpu() => LinearParityOn(() => new CudaBackend(0, PtxDirectory()), "CUDA");

    [ErnieVulkanFact]
    public void Vulkan_Fp8LinearWithMultipleBlocks_MatchesCpu() => LinearParityOn(() => new VulkanBackend(), "Vulkan");

    private static void LinearParityOn(Func<IBackend> open, string device)
    {
        const int n = 256, k = 384;
        Random rng = new(11);
        using Tensor raw = new(new TensorShape(n, k), DType.F8E4M3);
        Span<byte> bytes = raw.AsSpan<byte>();
        for (int i = 0; i < bytes.Length; i++)
        {
            byte v = (byte)rng.Next(256);
            if ((v & 0x7F) == 0x7F) v &= 0xFE;
            bytes[i] = v;
        }
        using Tensor scale = new(new TensorShape(2, 3), DType.F32);
        Span<float> s = scale.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(0.25 + rng.NextDouble());
        raw.QuantInfo = new QuantWeightInfo { Format = "fp8_block_scale_inv", Fp8BlockScaleInv = scale, Fp8BlockSize = 128 };
        using Tensor input = new(new TensorShape(1, k), DType.F32);
        Span<float> x = input.AsSpan<float>();
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        using CpuBackend cpu = new();
        using Tensor want = new(new TensorShape(1, n), DType.F32);
        cpu.Linear(want, input, raw, null);
        using IBackend gpu = open();
        using Tensor got = new(new TensorShape(1, n), DType.F32);
        gpu.Linear(got, input, raw, null);
        AssertClose(want.AsReadOnlySpan<float>().ToArray(), got.AsReadOnlySpan<float>().ToArray(), device);
    }

    // Compares the device's logits with the CPU's for the same checkpoint, weights and tokens.
    private void ParityOn(Func<IBackend> open, string device, bool resident, Ernie5SyntheticCheckpoint.Storage storage = Ernie5SyntheticCheckpoint.Storage.F32,
        IReadOnlyDictionary<string, string>? overrides = null)
    {
        string dir = Path.Combine(_root, Guid.NewGuid().ToString("N"));
        Ernie5SyntheticCheckpoint.Write(dir, storage: storage, configOverrides: overrides);
        using CpuBackend cpu = new();
        float[] want = Logits(dir, cpu, resident);
        using IBackend gpu = open();
        AssertClose(want, Logits(dir, gpu, resident), device);
    }

    private string WriteCheckpoint(Ernie5SyntheticCheckpoint.Storage storage)
    {
        string dir = Path.Combine(_root, Guid.NewGuid().ToString("N"));
        Ernie5SyntheticCheckpoint.Write(dir, storage: storage);
        return dir;
    }

    // A prefill chunk with every row's logits, then one decode step on the same state; both are compared, not just the last token.
    internal static float[] Logits(string dir, IBackend backend, bool resident)
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

    // Tolerance for both the F32 and the FP8 checkpoints: each side runs the same weights, so only accumulation order differs.
    internal static void AssertClose(float[] want, float[] got, string device)
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
    internal static string PtxDirectory()
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
