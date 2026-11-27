using HartsyInference.Core.Backends.Cpu;
using HartsyInference.LLM.Tests.Parity;
using HartsyInference.LLM.Ssm;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>The tiny SenseNova-Loop GGUFs on the Vulkan backend: the device Gated DeltaNet and Linear on the F32 file, and the quantized file through
/// the device quant GEMV, each compared with the CPU HF route. mHC runs on the host between device layers, as on CUDA.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class SenseNovaLoopVulkanTests
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

    [VulkanFact]
    public void TinyF32Gguf_OnVulkan_MatchesTheCpuHuggingFaceRoute()
    {
        (Dictionary<string, float[]> raw, Dictionary<string, Core.Tensors.Tensor> hfWeights) = SenseNovaLoopTests.TinyWeights(11, Streams);
        string json = SenseNovaLoopTests.TinyConfig(Streams);
        using CpuBackend cpu = new();
        using Qwen35Model reference = Qwen35Model.FromHuggingFace(hfWeights, Qwen35HfConfig.FromJson(json), 32, loop: SenseNovaLoopConfig.FromJson(json));
        float[] want = reference.ForwardHiddenStates(cpu, Ids);

        string path = Path.Combine(Path.GetTempPath(), $"sensenova_vk_{Guid.NewGuid():N}.gguf");
        try
        {
            SenseNovaLoopGgufTests.WriteTinyGguf(path, raw, quantize: false);
            using VulkanBackend vk = new();
            using Qwen35Model model = Qwen35Model.Load(path);
            vk.PreloadWeights(model.EnumerateWeights());
            double rel = RelRms(want, model.ForwardHiddenStates(vk, Ids));
            Assert.True(rel < 1e-3, $"F32 GGUF on Vulkan vs CPU HF route: relRMS {rel}");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [VulkanFact]
    public void TinyQ8Gguf_OnVulkan_RunsFiniteAndTracksTheF32File()
    {
        (Dictionary<string, float[]> raw, _) = SenseNovaLoopTests.TinyWeights(11, Streams);
        string f32Path = Path.Combine(Path.GetTempPath(), $"sensenova_vkq_f32_{Guid.NewGuid():N}.gguf");
        string q8Path = Path.Combine(Path.GetTempPath(), $"sensenova_vkq_q8_{Guid.NewGuid():N}.gguf");
        try
        {
            SenseNovaLoopGgufTests.WriteTinyGguf(f32Path, raw, quantize: false);
            SenseNovaLoopGgufTests.WriteTinyGguf(q8Path, raw, quantize: true);
            using CpuBackend cpu = new();
            float[] want;
            using (Qwen35Model f32 = Qwen35Model.Load(f32Path))
                want = f32.ForwardHiddenStates(cpu, Ids);
            using VulkanBackend vk = new();
            using Qwen35Model q8 = Qwen35Model.Load(q8Path);
            vk.PreloadWeights(q8.EnumerateWeights());
            float[] got = q8.ForwardHiddenStates(vk, Ids);
            Assert.All(got, v => Assert.True(float.IsFinite(v)));
            // Not a tight gate: Q8_0 rounding on the device is amplified by the tiny random model, as the CUDA Q8 test notes.
            double rel = RelRms(want, got);
            Assert.True(rel < 0.5, $"Q8_0 on Vulkan (device quant GEMV) vs the F32 file on the CPU: relRMS {rel}");
        }
        finally
        {
            File.Delete(f32Path);
            File.Delete(q8Path);
        }
    }
}
