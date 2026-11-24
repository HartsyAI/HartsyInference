using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ssm;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>A tiny random SenseNova-Loop trunk run through the checkpoint's own <c>modeling_sensenova_loop.py</c>
/// (<c>tools/sensenova_loop/backbone_reference.py</c>, transformers 5.18.0) must match the port built from the same weights.
/// Skipped (not passed) until <c>SENSENOVA_LOOP_TINY_EXPECTED</c> names the dumper's <c>sensenova_loop_tiny_expected.safetensors</c>.</summary>
public sealed unsafe class SenseNovaLoopHuggingFaceParityTests
{
    [RealWeightsFact("SENSENOVA_LOOP_TINY_EXPECTED")]
    public void HiddenStates_MatchTransformers()
    {
        string dir = Path.GetDirectoryName(Environment.GetEnvironmentVariable("SENSENOVA_LOOP_TINY_EXPECTED"))!;
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(dir, "sensenova_loop_tiny.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in loader.Descriptors.Keys) weights[name] = loader.GetTensor(name);
        using SafeTensorsLoader expectedLoader = new();
        expectedLoader.Load(Path.Combine(dir, "sensenova_loop_tiny_expected.safetensors"));
        Tensor want = expectedLoader.GetTensor("hidden"), stepped = expectedLoader.GetTensor("stepped_last"), idTensor = expectedLoader.GetTensor("input_ids");
        int[] ids = new ReadOnlySpan<int>((void*)idTensor.DataPointer, (int)idTensor.ElementCount).ToArray();
        string json = File.ReadAllText(Path.Combine(dir, "sensenova_loop_tiny_config.json"));
        SenseNovaLoopConfig loop = SenseNovaLoopConfig.FromJson(json)!;
        Qwen35HfConfig cfg = Qwen35HfConfig.FromJson(json);

        using CpuBackend backend = new();
        using Qwen35Model model = Qwen35Model.FromHuggingFace(weights, cfg, maxSequenceLength: 64, loop: loop);
        float[] got = model.ForwardHiddenStates(backend, ids);

        ReadOnlySpan<float> expected = new((void*)want.DataPointer, (int)want.ElementCount);
        Assert.Equal(expected.Length, got.Length);
        double num = 0, den = 0;
        for (int i = 0; i < got.Length; i++)
        {
            num += (expected[i] - got[i]) * (double)(expected[i] - got[i]);
            den += (double)expected[i] * expected[i];
        }
        Assert.True(Math.Sqrt(num / den) < 1e-4, $"hidden states relRMS {Math.Sqrt(num / den)}");
        // The reference's own cache path (prefill plus steps) agrees with its full forward; so must ours.
        int hidden = cfg.HiddenSize;
        ReadOnlySpan<float> last = new((void*)stepped.DataPointer, hidden);
        double worst = 0;
        for (int c = 0; c < hidden; c++) worst = Math.Max(worst, Math.Abs(last[c] - got[(ids.Length - 1) * hidden + c]));
        Assert.True(worst < 2e-4, $"last position vs the reference cache path: max |d| {worst}");
    }
}
