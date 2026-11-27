using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ssm;
using Xunit;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>The quantized route on the CPU. The Qwen3.5 loader keeps quantized weights packed for a backend that multiplies by them; the CPU does
/// not, so the engine loads the tiny Q8_0 file with every weight expanded to F32 and the forward runs on the CPU, tracking the F32 file.</summary>
public sealed class SenseNovaLoopCpuQuantTests
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

    [Fact]
    public void TinyQ8Gguf_DequantizedAtLoad_RunsOnCpuAndTracksTheF32File()
    {
        (Dictionary<string, float[]> raw, _) = SenseNovaLoopTests.TinyWeights(11, Streams);
        string f32Path = Path.Combine(Path.GetTempPath(), $"sensenova_cpuq_f32_{Guid.NewGuid():N}.gguf");
        string q8Path = Path.Combine(Path.GetTempPath(), $"sensenova_cpuq_q8_{Guid.NewGuid():N}.gguf");
        try
        {
            SenseNovaLoopGgufTests.WriteTinyGguf(f32Path, raw, quantize: false);
            SenseNovaLoopGgufTests.WriteTinyGguf(q8Path, raw, quantize: true);
            using CpuBackend cpu = new();
            float[] want;
            using (Qwen35Model f32 = Qwen35Model.Load(f32Path))
                want = f32.ForwardHiddenStates(cpu, Ids);
            using Qwen35Model q8 = Qwen35Model.Load(q8Path, dequantizeToF32: true);
            float[] got = q8.ForwardHiddenStates(cpu, Ids);
            Assert.All(got, v => Assert.True(float.IsFinite(v)));
            // Not a tight gate: the tiny random model amplifies Q8_0 rounding (0.08 to 0.17 measured on the device), as the CUDA Q8 test notes.
            double rel = RelRms(want, got);
            Assert.True(rel < 0.5, $"Q8_0 dequantized at load vs the F32 file on the CPU: relRMS {rel}");
        }
        finally
        {
            File.Delete(f32Path);
            File.Delete(q8Path);
        }
    }

    [Fact]
    public void TinyQ8Gguf_LoadedPacked_IsRefusedOnTheCpuAtTheFirstLinear()
    {
        // The negative control: without the expansion the CPU backend refuses the packed weights, so the engine must not take that route on the CPU.
        (Dictionary<string, float[]> raw, _) = SenseNovaLoopTests.TinyWeights(11, Streams);
        string q8Path = Path.Combine(Path.GetTempPath(), $"sensenova_cpuq_packed_{Guid.NewGuid():N}.gguf");
        try
        {
            SenseNovaLoopGgufTests.WriteTinyGguf(q8Path, raw, quantize: true);
            using CpuBackend cpu = new();
            using Qwen35Model packed = Qwen35Model.Load(q8Path);
            Assert.ThrowsAny<Exception>(() => packed.ForwardHiddenStates(cpu, Ids));
        }
        finally
        {
            File.Delete(q8Path);
        }
    }
}
