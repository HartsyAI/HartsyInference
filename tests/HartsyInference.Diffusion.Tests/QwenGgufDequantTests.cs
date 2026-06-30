using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelHandler.Gguf;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Validates the Q4_K dequant (both the CPU <see cref="GgufDequantizer"/> and the CUDA
/// <c>CastToF32</c> GPU kernel) against a gguf-python reference dequant of the SAME tensor from the
/// Qwen-Image GGUF. If either diverges, the GGUF weights reaching the transformer are wrong → that is the
/// root cause of garbage output, independent of any pipeline logic. Reference produced by:
/// <c>gguf.quants.dequantize(tensor.data, Q4_K)</c> saved as raw little-endian f32 at $QWEN_Q4K_REF.</summary>
public unsafe class QwenGgufDequantTests
{
    private readonly ITestOutputHelper _output;
    public QwenGgufDequantTests(ITestOutputHelper output) => _output = output;

    private const string TensorName = "transformer_blocks.9.txt_mod.1.weight";

    [Fact]
    public void Q4K_Dequant_Matches_GgufPython_CpuAndGpu()
    {
        string ggufPath = TestPaths.QwenImage.V1Gguf;
        string refPath = Environment.GetEnvironmentVariable("QWEN_Q4K_REF")
            ?? "/tmp/claude-1000/-home-hartsy-Desktop-HartsyInference/e2a786ca-1f44-4a22-9e05-07d1e3771c4d/scratchpad/q4k_ref.f32";
        if (!File.Exists(ggufPath)) { _output.WriteLine($"SKIPPED: GGUF not found {ggufPath}"); return; }
        if (!File.Exists(refPath)) { _output.WriteLine($"SKIPPED: ref not found {refPath}"); return; }

        byte[] refBytes = File.ReadAllBytes(refPath);
        int n = refBytes.Length / 4;
        float[] reference = new float[n];
        Buffer.BlockCopy(refBytes, 0, reference, 0, refBytes.Length);
        _output.WriteLine($"reference n={n} first4=[{reference[0]:F6},{reference[1]:F6},{reference[2]:F6},{reference[3]:F6}]");

        using GgufLoader loader = new();
        loader.Load(ggufPath);
        Tensor q = loader.GetTensor(TensorName);
        _output.WriteLine($"GGUF tensor {TensorName}: dtype={q.DType} shape={q.Shape} elems={q.ElementCount}");
        Assert.Equal(n, (int)q.ElementCount);

        // ── CPU dequant ──
        Tensor cpu = GgufDequantizer.Dequantize(q, DType.F32);
        ReadOnlySpan<float> cpuS = cpu.AsReadOnlySpan<float>();
        (float maxCpu, int argCpu) = MaxAbsDiff(cpuS, reference);
        _output.WriteLine($"CPU dequant: first4=[{cpuS[0]:F6},{cpuS[1]:F6},{cpuS[2]:F6},{cpuS[3]:F6}]  maxAbsDiff={maxCpu:E4} @ {argCpu}");

        // ── GPU dequant (the path the failing generation used) ──
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(QwenGgufDequantTests).Assembly.Location)!, "Ptx");
        float maxGpu = float.NaN;
        if (Directory.Exists(ptxDir))
        {
            try
            {
                using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
                Tensor gpuOut = backend.DequantizeToF32(q);   // real generation dequant path (CastOnGpu → LaunchDequantQ4_KToF16)
                backend.Sync();
                ReadOnlySpan<float> gpuS = gpuOut.AsReadOnlySpan<float>();
                (maxGpu, int argGpu) = MaxAbsDiff(gpuS, reference);
                _output.WriteLine($"GPU dequant: first4=[{gpuS[0]:F6},{gpuS[1]:F6},{gpuS[2]:F6},{gpuS[3]:F6}]  maxAbsDiff={maxGpu:E4} @ {argGpu}");
            }
            catch (Exception e) { _output.WriteLine($"GPU dequant FAILED: {e.GetType().Name}: {e.Message}"); }
        }
        else _output.WriteLine("GPU dequant skipped (no PTX dir)");

        Assert.True(maxCpu < 1e-3f, $"CPU Q4_K dequant diverges from gguf-python by {maxCpu}");
        if (!float.IsNaN(maxGpu))
            Assert.True(maxGpu < 1e-2f, $"GPU Q4_K dequant diverges from gguf-python by {maxGpu}");
    }

    private static (float max, int arg) MaxAbsDiff(ReadOnlySpan<float> a, float[] b)
    {
        float m = 0; int arg = 0;
        for (int i = 0; i < b.Length; i++) { float d = MathF.Abs(a[i] - b[i]); if (d > m) { m = d; arg = i; } }
        return (m, arg);
    }
}
