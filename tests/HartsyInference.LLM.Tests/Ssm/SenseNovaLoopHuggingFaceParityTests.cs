using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ssm;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>A tiny random SenseNova-Loop trunk run through the checkpoint's own <c>modeling_sensenova_loop.py</c>
/// (<c>tools/sensenova_loop/backbone_reference.py</c>, transformers 5.18.0) must match the port built from the same weights.
/// The dumper's output (1.5 MB, made 2026-11-24 in the CPU container: torch 2.7.1, transformers 5.24.0) is committed under
/// <c>Fixtures/SenseNovaLoop</c>, so this runs everywhere.</summary>
public sealed unsafe class SenseNovaLoopHuggingFaceParityTests
{
    [Fact]
    public void HiddenStates_MatchTransformers()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "SenseNovaLoop");
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
        Assert.True(Math.Sqrt(num / den) < 2e-5, $"hidden states relRMS {Math.Sqrt(num / den)}");
        // The reference's own cache path (prefill plus steps) agrees with its full forward; so must ours.
        int hidden = cfg.HiddenSize;
        ReadOnlySpan<float> last = new((void*)stepped.DataPointer, hidden);
        double worst = 0;
        for (int c = 0; c < hidden; c++) worst = Math.Max(worst, Math.Abs(last[c] - got[(ids.Length - 1) * hidden + c]));
        Assert.True(worst < 2e-4, $"last position vs the reference cache path: max |d| {worst}");
    }

    [Fact]
    public void ImagePrompt_PositionsAndHiddenStatesMatchTransformers()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "SenseNovaLoop");
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(dir, "sensenova_loop_tiny.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in loader.Descriptors.Keys) weights[name] = loader.GetTensor(name);
        using SafeTensorsLoader expectedLoader = new();
        expectedLoader.Load(Path.Combine(dir, "sensenova_loop_tiny_expected.safetensors"));
        Tensor idTensor = expectedLoader.GetTensor("mm_input_ids"), positionTensor = expectedLoader.GetTensor("mm_positions"), want = expectedLoader.GetTensor("mm_hidden");
        int[] ids = new ReadOnlySpan<int>((void*)idTensor.DataPointer, (int)idTensor.ElementCount).ToArray();
        int[] referencePositions = new ReadOnlySpan<int>((void*)positionTensor.DataPointer, (int)positionTensor.ElementCount).ToArray();
        string json = File.ReadAllText(Path.Combine(dir, "sensenova_loop_tiny_config.json"));
        Qwen35HfConfig cfg = Qwen35HfConfig.FromJson(json);
        const int imageToken = 2;

        // Transformers' own get_rope_index versus the engine's position builder, over [text, text, start, 2x2 image, end, text, text, text].
        int[] positions = MRopePositions.Build(ids, imageToken, [new MRopeImageGrid(1, 2, 2)], out _);
        Assert.Equal(referencePositions, positions);

        using CpuBackend backend = new();
        using Qwen35Model model = Qwen35Model.FromHuggingFace(weights, cfg, maxSequenceLength: 64, loop: SenseNovaLoopConfig.FromJson(json));
        int seq = ids.Length, hidden = cfg.HiddenSize;
        using Tensor embeds = new(new TensorShape(1, seq, hidden), DType.F32);
        model.EmbedLookup(embeds, ids);
        byte[] pairAxes = MRopePairAxes.Build([.. model.MropeSection], model.RotaryDim / 2, model.MropeInterleaved ? MRopeLayout.Interleaved : MRopeLayout.Sectioned);
        (double[] invFreq, double scaling) = RopeFrequencyBuilder.Build(model.RotaryDim, model.RopeTheta, model.RopeScaling, 64);
        (Tensor cos, Tensor sin) = MRopeTable.BuildTensors(referencePositions, seq, 0, seq, invFreq, scaling, pairAxes, model.AttnHeadDim, model.RotaryDim);
        using (cos)
        using (sin)
        {
            model.ResetState();
            using Tensor got = model.ForwardEmbedsHidden(backend, embeds, seq, cos, sin);
            ReadOnlySpan<float> expected = new((void*)want.DataPointer, (int)want.ElementCount), actual = got.AsReadOnlySpan<float>();
            double num = 0, den = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                num += (expected[i] - (double)actual[i]) * (expected[i] - (double)actual[i]);
                den += (double)expected[i] * expected[i];
            }
            Assert.True(Math.Sqrt(num / den) < 2e-5, $"image prompt hidden states relRMS {Math.Sqrt(num / den)}");
        }
    }
}
