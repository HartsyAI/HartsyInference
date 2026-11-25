using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Ssm;
using HartsyInference.LLM.Tests.Parity;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>The tiny SenseNova-Loop GGUFs on the CUDA backend: the device Gated DeltaNet kernels at a value-to-key head ratio of 2, device Linear on
/// row-sliced and permuted quantized tensors and on the F16 <c>ssm_out</c>, and the host mHC blocks between device layers. Needs a free card: run
/// under the GPU lock, one suite at a time.</summary>
/// <remarks>Measured 2026-11-24 on the RTX 3060: a tiny random model amplifies the CUDA backend's own Q8_0 rounding (activations are quantized too),
/// so a Q8_0 model is not compared to F32 within a tight gate. The GGUF route is compared with the HF route holding the same Q8_0 rows on the same
/// backend, which isolates what this port adds (slicing, un-tiling, aliasing) from the kernels.</remarks>
[Trait("Category", "GpuIntegration")]
public sealed class SenseNovaLoopGpuTests
{
    private const int Streams = 2;
    private static readonly int[] Ids = [3, 17, 5, 29, 8, 1, 22];

    private static double RelRms(float[] want, float[] got)
    {
        double num = 0, den = 0;
        for (int i = 0; i < want.Length; i++)
        {
            num += (want[i] - (double)got[i]) * (want[i] - (double)got[i]);
            den += (double)want[i] * want[i];
        }
        return Math.Sqrt(num / den);
    }

    [CudaFact]
    public void TinyF32Gguf_OnCuda_MatchesTheCpuHuggingFaceRoute()
    {
        (Dictionary<string, float[]> raw, Dictionary<string, Tensor> hfWeights) = SenseNovaLoopTests.TinyWeights(11, Streams);
        string json = SenseNovaLoopTests.TinyConfig(Streams);
        using CpuBackend cpu = new();
        using Qwen35Model reference = Qwen35Model.FromHuggingFace(hfWeights, Qwen35HfConfig.FromJson(json), 32, loop: SenseNovaLoopConfig.FromJson(json));
        float[] want = reference.ForwardHiddenStates(cpu, Ids);

        string path = Path.Combine(Path.GetTempPath(), $"sensenova_loop_gpu_{Guid.NewGuid():N}.gguf");
        try
        {
            SenseNovaLoopGgufTests.WriteTinyGguf(path, raw, quantize: false);
            using CudaBackend cuda = new(0, CudaPtxDirectory.Resolve());
            using Qwen35Model model = Qwen35Model.Load(path);
            cuda.PreloadWeights(model.EnumerateWeights());
            double rel = RelRms(want, model.ForwardHiddenStates(cuda, Ids));
            Assert.True(rel < 1e-4, $"F32 GGUF on CUDA vs CPU HF route: relRMS {rel}");   // summation order only
        }
        finally
        {
            File.Delete(path);
        }
    }

    [CudaFact]
    public void TinyQ8Gguf_OnCuda_EqualsTheHuggingFaceRouteHoldingTheSameRows()
    {
        (Dictionary<string, float[]> raw, Dictionary<string, Tensor> hfWeights) = SenseNovaLoopTests.TinyWeights(11, Streams);
        string json = SenseNovaLoopTests.TinyConfig(Streams);
        // The same Q8_0 rows the file holds (row-wise quantization commutes with the converter's row moves), ssm_out left F32 in both.
        Dictionary<string, Tensor> quantizedHf = new();
        foreach ((string name, Tensor like) in hfWeights)
        {
            bool rows = name.Contains("in_proj_") || name.Contains("gate_proj") || name.Contains("up_proj") || name.EndsWith("q_proj.weight")
                || name.EndsWith("k_proj.weight") || name.EndsWith("v_proj.weight") || name.EndsWith("o_proj.weight");
            quantizedHf[name] = rows && like.Shape.Rank == 2 && like.Shape[1] % 32 == 0 ? GgufQuantizer.Quantize(like, DType.Q8_0) : like;
        }
        string path = Path.Combine(Path.GetTempPath(), $"sensenova_loop_gpu_{Guid.NewGuid():N}.gguf");
        try
        {
            SenseNovaLoopGgufTests.WriteTinyGguf(path, raw, quantize: true, quantizeSsmOut: false);
            using CudaBackend cuda = new(0, CudaPtxDirectory.Resolve());
            using Qwen35Model fromHf = Qwen35Model.FromHuggingFace(quantizedHf, Qwen35HfConfig.FromJson(json), 32, loop: SenseNovaLoopConfig.FromJson(json));
            cuda.PreloadWeights(fromHf.EnumerateWeights());
            float[] want = fromHf.ForwardHiddenStates(cuda, Ids);
            using Qwen35Model fromGguf = Qwen35Model.Load(path);
            cuda.PreloadWeights(fromGguf.EnumerateWeights());
            double rel = RelRms(want, fromGguf.ForwardHiddenStates(cuda, Ids));
            Assert.True(rel < 1e-6, $"Q8_0 GGUF vs HF route with the same rows, both on CUDA: relRMS {rel}");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [CudaFact]
    public void TinyQ8Gguf_WithQuantizedSsmOut_RunsOnCuda()
    {
        (Dictionary<string, float[]> raw, Dictionary<string, Tensor> hfWeights) = SenseNovaLoopTests.TinyWeights(11, Streams);
        string json = SenseNovaLoopTests.TinyConfig(Streams);
        using CpuBackend cpu = new();
        using Qwen35Model reference = Qwen35Model.FromHuggingFace(hfWeights, Qwen35HfConfig.FromJson(json), 32, loop: SenseNovaLoopConfig.FromJson(json));
        float[] want = reference.ForwardHiddenStates(cpu, Ids);
        string path = Path.Combine(Path.GetTempPath(), $"sensenova_loop_gpu_{Guid.NewGuid():N}.gguf");
        try
        {
            SenseNovaLoopGgufTests.WriteTinyGguf(path, raw, quantize: true);
            using CudaBackend cuda = new(0, CudaPtxDirectory.Resolve());
            using Qwen35Model model = Qwen35Model.Load(path);
            cuda.PreloadWeights(model.EnumerateWeights());
            float[] got = model.ForwardHiddenStates(cuda, Ids);
            Assert.All(got, v => Assert.True(float.IsFinite(v)));
            // Not a tight gate: Q8_0 rounding on the device is amplified by this tiny random model (0.08 to 0.17 measured against F32 with qkv alone).
            Assert.True(RelRms(want, got) < 0.5);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
