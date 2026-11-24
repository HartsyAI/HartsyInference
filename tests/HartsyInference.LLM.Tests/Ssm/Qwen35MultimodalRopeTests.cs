using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ssm;
using Xunit;
using static HartsyInference.LLM.Tests.Ssm.SenseNovaLoopReference;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>Image input on the Qwen3.5 trunk: the placeholder run takes grid positions and the text after it resumes at <c>start + max(T, H, W)</c>,
/// below the sequence index. A prefill with those rows followed by single-token decode (<see cref="Qwen35Model.RopePositionOffset"/>) must
/// equal one pass over the whole sequence with the same rows, on the looped mHC model (the only trunk here with an image-capable checkpoint).</summary>
public sealed unsafe class Qwen35MultimodalRopeTests
{
    private const int Placeholder = -1;

    [Fact]
    public void PrefillThenDecode_FollowsTheImageGridPositions()
    {
        const int streams = 2;
        (_, Dictionary<string, Tensor> weights) = SenseNovaLoopTests.TinyWeights(17, streams);
        string json = SenseNovaLoopTests.TinyConfig(streams);
        using Qwen35Model model = Qwen35Model.FromHuggingFace(weights, Qwen35HfConfig.FromJson(json), 32, loop: SenseNovaLoopConfig.FromJson(json));
        using CpuBackend backend = new();

        // [text] [2x2 image grid] [text] [text]: placeholder embeddings are ordinary token rows here.
        int[] ids = [5, 11, 12, 13, 14, 20, 9];
        int[] layout = [0, Placeholder, Placeholder, Placeholder, Placeholder, 0, 0];
        MRopeImageGrid[] grids = [new(1, 2, 2)];
        int[] positions = MRopePositions.Build(layout, Placeholder, grids, out int nextAfterAll);
        Assert.Equal(new[] { 0, 1, 1, 1, 1, 3, 4 }, positions[..7]);   // temporal axis: text 0, image start 1, text resumes at 3
        Assert.Equal(new[] { 0, 1, 1, 2, 2, 3, 4 }, positions[7..14]);   // height
        Assert.Equal(5, nextAfterAll);

        using Tensor embeds = new(new TensorShape(1, ids.Length, model.DModel), DType.F32);
        model.EmbedLookup(embeds, ids);
        int rotaryDim = model.RotaryDim;
        byte[] pairAxes = MRopePairAxes.Build([.. model.MropeSection], rotaryDim / 2, model.MropeInterleaved ? MRopeLayout.Interleaved : MRopeLayout.Sectioned);
        (double[] invFreq, double scaling) = RopeFrequencyBuilder.Build(rotaryDim, model.RopeTheta, model.RopeScaling, 64);

        // One pass over all seven rows.
        (Tensor cosAll, Tensor sinAll) = MRopeTable.BuildTensors(positions, 7, 0, 7, invFreq, scaling, pairAxes, model.AttnHeadDim, rotaryDim);
        model.ResetState();
        float[] want = model.ForwardEmbedsLastLogits(backend, embeds, 7, cosAll, sinAll);
        cosAll.Dispose();
        sinAll.Dispose();

        // Prefill six rows, then decode the seventh through the offset.
        int[] prefixPositions = MRopePositions.Build(layout[..6], Placeholder, grids, out int nextAfterPrefix);
        (Tensor cosPre, Tensor sinPre) = MRopeTable.BuildTensors(prefixPositions, 6, 0, 6, invFreq, scaling, pairAxes, model.AttnHeadDim, rotaryDim);
        using Tensor prefix = new(new TensorShape(1, 6, model.DModel), DType.F32);
        new ReadOnlySpan<float>((void*)embeds.DataPointer, 6 * model.DModel).CopyTo(prefix.AsSpan<float>());
        model.ResetState();
        model.ForwardEmbedsLastLogits(backend, prefix, 6, cosPre, sinPre);
        cosPre.Dispose();
        sinPre.Dispose();
        model.RopePositionOffset = nextAfterPrefix - 6;
        float[] got = model.ForwardLastLogits(backend, [ids[6]]);
        Assert.True(RelRms(want, got) < 1e-5, $"decode after an image vs one pass: relRMS {RelRms(want, got)}");

        // The same decode without the offset rotates by the sequence index and must differ: the check has power.
        (Tensor cosPre2, Tensor sinPre2) = MRopeTable.BuildTensors(prefixPositions, 6, 0, 6, invFreq, scaling, pairAxes, model.AttnHeadDim, rotaryDim);
        model.ResetState();
        model.ForwardEmbedsLastLogits(backend, prefix, 6, cosPre2, sinPre2);
        cosPre2.Dispose();
        sinPre2.Dispose();
        float[] wrong = model.ForwardLastLogits(backend, [ids[6]]);
        Assert.True(RelRms(want, wrong) > 1e-3, $"without the offset the logits should differ, relRMS {RelRms(want, wrong)}");
    }

    private static double RelRms(float[] want, float[] got)
    {
        double num = 0, den = 0;
        for (int i = 0; i < want.Length; i++)
        {
            num += (want[i] - (double)got[i]) * (want[i] - (double)got[i]);
            den += (double)want[i] * want[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }
}
