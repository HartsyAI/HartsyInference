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

    /// <summary>The bound on a logit's difference over max(1, |CPU logit|).</summary>
    private const float Tolerance = 1e-3f;

    /// <summary>The bound on the decode step of the real-dimension checkpoint, which is sensitive to rounding. On the CPU alone, a run with
    /// AVX2 disabled differs from the AVX2 run by up to 6.5e-4 in the decode step (1.1e-4 in the prefill rows), and moving every F32 weight of
    /// the dequantized twin by one ulp moves it by up to 4.2e-4 (prefill 8.0e-5); both peak at logit 22962. On an RTX 4090 at
    /// <c>cacee5b7b</c>, against that host's CPU: CUDA with exact GEMMs 9.5e-4 in the decode step, at logit 22962 (prefill 6.2e-5); Vulkan
    /// with F16 GEMM off 1.005e-3 (prefill 5.9e-5).</summary>
    private const float RealDimensionDecodeTolerance = 3e-3f;

    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-ernie5-gpu-" + Guid.NewGuid().ToString("N"));

    private readonly Xunit.Abstractions.ITestOutputHelper _output;

    public Ernie5ModelGpuParityTests(Xunit.Abstractions.ITestOutputHelper output) => _output = output;

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
    // path, is what this comparison measures. On both devices the decode step has its own bound, RealDimensionDecodeTolerance.
    [ErnieCudaFact]
    public void Cuda_RealDimensionsFp8_LogitsMatchCpu() => ParityOn(() => new CudaBackend(0, PtxDirectory()) { HighPrecisionGemm = true }, "CUDA",
        resident: false, Ernie5SyntheticCheckpoint.Storage.Fp8, Ernie5SyntheticCheckpoint.RealDimensionOverrides, RealDimensionDecodeTolerance);

    // F16 GEMM off, the Vulkan counterpart of HighPrecisionGemm above: F16 GEMM would round the inputs to F16, about 1e-3 off by itself.
    [ErnieVulkanFact]
    public void Vulkan_RealDimensionsFp8_LogitsMatchCpu() => ParityOn(() => new VulkanBackend { EnableF16Gemm = false }, "Vulkan", resident: false,
        Ernie5SyntheticCheckpoint.Storage.Fp8, Ernie5SyntheticCheckpoint.RealDimensionOverrides, RealDimensionDecodeTolerance);

    // The engine's Text service end to end on the device: the slot's backend is the requested GPU (resident layers), and a request generates on the
    // pipeline and, with continuous batching on, on the scheduler under the device gate. The tokens must match the CPU run of the same request.
    // FP8 is the real checkpoint's storage, whose resident preload and per-call expert upload the service turns on.
    [ErnieCudaFact]
    public Task Cuda_TextService_LoadsOnTheDeviceAndMatchesCpu() => TextServiceOn("cuda:0", Ernie5SyntheticCheckpoint.Storage.F32);

    [ErnieCudaFact]
    public Task Cuda_TextService_Fp8_LoadsOnTheDeviceAndMatchesCpu() => TextServiceOn("cuda:0", Ernie5SyntheticCheckpoint.Storage.Fp8);

    [ErnieVulkanFact]
    public Task Vulkan_TextService_LoadsOnTheDeviceAndMatchesCpu() => TextServiceOn("vulkan:0", Ernie5SyntheticCheckpoint.Storage.F32);

    [ErnieVulkanFact]
    public Task Vulkan_TextService_Fp8_LoadsOnTheDeviceAndMatchesCpu() => TextServiceOn("vulkan:0", Ernie5SyntheticCheckpoint.Storage.Fp8);

    private async Task TextServiceOn(string device, Ernie5SyntheticCheckpoint.Storage storage)
    {
        string dir = Path.Combine(_root, $"service-{storage}-" + device.Replace(':', '-'));
        Ernie5SyntheticCheckpoint.Write(dir, storage: storage);
        Ernie5TextPathTests.ServiceRun cpu = await Ernie5TextPathTests.GenerateThroughService(dir, "cpu", continuousBatching: false);
        foreach (bool batching in new[] { false, true })
        {
            Ernie5TextPathTests.ServiceRun run = await Ernie5TextPathTests.GenerateThroughService(dir, device, batching);
            Assert.True(run.Device.IsGpu, $"{device} loaded on {run.Device}.");
            Assert.Equal(batching, run.Scheduled);
            Assert.Equal(cpu.Result.Text, run.Result.Text);
        }
    }

    // One FP8 linear whose weight spans two row blocks and three column blocks with distinct scales, on the device and the CPU.
    // Two stages on one device (two backends, so each holds its own weight cache), every weight preloaded: the split and the expert preload must
    // give the CPU's logits, and no forward may read a routed expert from the checkpoint.
    [ErnieCudaFact]
    public void Cuda_SyntheticErnieFp8_TwoStagesWithEveryWeightPreloaded_LogitsMatchCpu()
    {
        string dir = WriteCheckpoint(Ernie5SyntheticCheckpoint.Storage.Fp8);
        using CpuBackend cpu = new();
        float[] want = Logits(dir, cpu, resident: true);
        using CudaBackend first = new(0, PtxDirectory());
        using CudaBackend second = new(0, PtxDirectory());
        int layers = Ernie5Config.Load(Path.Combine(dir, "config.json")).NumHiddenLayers;
        LLM.Transformer.LlmPlacement split = new([new LLM.Transformer.LlmStage(first, 0, 1), new LLM.Transformer.LlmStage(second, 1, layers)]);
        using Ernie5Model model = Ernie5Model.Load(dir, second, new Ernie5LoadOptions
        {
            MaxSequenceTokens = Tokens.Length + 1, ResidentWeights = true, Placement = split, PreloadExperts = true,
        });
        // Every routed expert is on one of the two devices' caches before the first forward.
        Assert.All(model.EnumerateExpertWeights(), t => Assert.True(first.IsWeightResident(t) || second.IsWeightResident(t), "a routed expert is not preloaded"));
        using Ernie5SequenceState state = model.CreateState(Tokens.Length + 1);
        float[] prefill = model.Forward(Tokens, state, allRows: true);
        float[] decode = model.Forward([Tokens[^1] + 1], state);
        AssertClose(want, [.. prefill, .. decode], "CUDA two-stage");
    }

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
        IReadOnlyDictionary<string, string>? overrides = null, float decodeTolerance = Tolerance)
    {
        string dir = Path.Combine(_root, Guid.NewGuid().ToString("N"));
        Ernie5SyntheticCheckpoint.Write(dir, storage: storage, configOverrides: overrides);
        using CpuBackend cpu = new();
        float[] want = Logits(dir, cpu, resident);
        using IBackend gpu = open();
        // One row of logits per prompt token, then the decode step's row.
        int decodeStart = want.Length / (Tokens.Length + 1) * Tokens.Length;
        _output.WriteLine(AssertClose(want, Logits(dir, gpu, resident), device, decodeStart, decodeTolerance));
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

    // Each side runs the same weights, so the sides differ by rounding. A logit before decodeStart is bounded by Tolerance, one from
    // there on by decodeTolerance. The returned summary, also the failure message, covers every logit, not only the first one past its bound.
    internal static string AssertClose(float[] want, float[] got, string device, int decodeStart = int.MaxValue,
        float decodeTolerance = Tolerance)
    {
        Assert.Equal(want.Length, got.Length);
        int split = Math.Min(decodeStart, want.Length), over = 0, first = -1, prefillAt = 0, decodeAt = split;
        float prefillWorst = 0f, decodeWorst = 0f;
        for (int i = 0; i < want.Length; i++)
        {
            Assert.True(float.IsFinite(got[i]), $"{device} logit {i} is {got[i]}");
            float difference = MathF.Abs(want[i] - got[i]) / MathF.Max(1f, MathF.Abs(want[i]));
            if (i < split && difference > prefillWorst) (prefillWorst, prefillAt) = (difference, i);
            if (i >= split && difference > decodeWorst) (decodeWorst, decodeAt) = (difference, i);
            // Negated, so a NaN from the CPU side counts as past the bound.
            if (!(difference <= (i < split ? Tolerance : decodeTolerance)))
            {
                over++;
                if (first < 0) first = i;
            }
        }
        string summary = $"{device}: largest relative difference {prefillWorst:G4} at logit {prefillAt} (bound {Tolerance:G2})";
        if (split < want.Length) summary += $"; decode step {decodeWorst:G4} at logit {decodeAt} (bound {decodeTolerance:G2})";
        if (over > 0) summary += $"; {over} of {want.Length} past the bound, first logit {first}: cpu {want[first]}, device {got[first]}";
        Assert.True(over == 0, summary);
        return summary;
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
