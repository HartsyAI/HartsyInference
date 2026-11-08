using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Models.MossVl;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Tests;

/// <summary>The text decoder's first three layers (self, self, gated cross-attention) on the real checkpoint, against the numpy
/// transcription in <c>tests/python-reference/dump_moss_vl_text.py</c>. <c>HARTSY_MOSS_VL_INDEX</c> is the checkpoint's index,
/// <c>HARTSY_MOSS_VL_REF</c> the vision dump's <c>merged.npy</c> (the image input), <c>HARTSY_MOSS_VL_TEXT_REF</c> the text dump's
/// <c>text_hidden.npy</c>; <c>cross_input.npy</c> and <c>prompt_ids.txt</c> sit beside it.</summary>
/// <remarks>The checkpoint's cross-attention gates are about 1e-2, so the cross layer moves the hidden state only slightly. The test
/// therefore also compares the layer's own contribution (its output minus its input), which a missing rotation or row mask would
/// change by most of its size.</remarks>
[Trait("Category", "Integration")]
public sealed class MossVlTextParityTests
{
    private const string CheckpointVariable = "HARTSY_MOSS_VL_INDEX";
    private const string VisionVariable = "HARTSY_MOSS_VL_REF";
    private const string TextVariable = "HARTSY_MOSS_VL_TEXT_REF";
    private const int Layers = 3;
    private const int CrossLayer = 2;
    private const int MergedHeight = 8, MergedWidth = 12;

    /// <summary>Largest allowed <c>max|diff| / max|ref|</c>. Both sides run F32 on the same widened weights.</summary>
    private const double RelativeTolerance = 1e-2;

    [RealWeightsFact(CheckpointVariable, VisionVariable, TextVariable)]
    public void DecoderLayersMatchTheReferenceAndTheCrossLayerDeltaMatches()
    {
        string checkpoint = Path.GetDirectoryName(Environment.GetEnvironmentVariable(CheckpointVariable)!)!;
        string mergedPath = Environment.GetEnvironmentVariable(VisionVariable)!;
        string textRefPath = Environment.GetEnvironmentVariable(TextVariable)!;
        string refDir = Path.GetDirectoryName(textRefPath)!;
        int[] ids = File.ReadAllText(Path.Combine(refDir, "prompt_ids.txt"))
            .Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();

        MossVlConfig config;
        using (JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(checkpoint, "config.json"))))
            config = MossVlConfig.FromJson(doc.RootElement);

        (long[] mergedShape, float[] mergedData) = NpyReader.ReadFloat32(mergedPath);
        (long[] refShape, float[] refHidden) = NpyReader.ReadFloat32(textRefPath);
        (_, float[] refCrossInput) = NpyReader.ReadFloat32(Path.Combine(refDir, "cross_input.npy"));
        Assert.Equal((long)ids.Length, refShape[0]);

        using MossVlTextDecoder decoder = MossVlTextDecoder.Open(checkpoint, config, Layers);
        using IBackend backend = new CpuBackend();

        using Tensor merged = new(new TensorShape(mergedShape[0], mergedShape[1]), DType.F32);
        mergedData.AsSpan().CopyTo(merged.AsSpan<float>());
        using Tensor vision = decoder.AppendSeparator(merged);

        MossVlPositionPlan plan = MossVlPositions.Plan(ids, config.ImageTokenId, [new MossVlImage(MergedHeight, MergedWidth)]);
        using MossVlRope rope = MossVlPositions.BuildRope(plan, config);
        using Tensor embeds = new(new TensorShape(1, ids.Length, config.Text.HiddenSize), DType.F32);
        decoder.Model.EmbedLookup(embeds, ids);
        using FixedKvCache cache = new(Layers, 1, config.Text.NumKvHeads, config.Text.HeadDim, ids.Length + 1);

        // Layers [0, CrossLayer) first, then the cross-attention layer alone on that state, so its contribution can be read off.
        using Tensor before = decoder.Model.ForwardEmbeds(backend, embeds, ids.Length, 0, cache, applyFinalNorm: false,
            endLayer: CrossLayer, crossStates: vision, crossLen: (int)vision.Shape[1], rope: rope.Rope);
        using Tensor after = decoder.Model.ForwardEmbeds(backend, before, ids.Length, 0, cache, applyFinalNorm: false,
            startLayer: CrossLayer, endLayer: CrossLayer + 1, crossStates: vision, crossLen: (int)vision.Shape[1], rope: rope.Rope,
            advanceCache: false);

        ReadOnlySpan<float> h1 = before.AsSpan<float>();
        ReadOnlySpan<float> h2 = after.AsSpan<float>();
        Assert.Equal(refCrossInput.Length, h1.Length);
        Assert.Equal(refHidden.Length, h2.Length);

        // The layers before the cross layer, and the full stack's output.
        Report("layers 0-1", h1, refCrossInput, out double preRelative);
        Assert.True(preRelative < RelativeTolerance, $"Layers 0-1 relative {preRelative:G6} >= {RelativeTolerance}.");
        Report("layers 0-2", h2, refHidden, out double fullRelative);
        Assert.True(fullRelative < RelativeTolerance, $"Layers 0-2 relative {fullRelative:G6} >= {RelativeTolerance}.");

        // The cross-attention layer's own contribution: output minus input, on both sides.
        float[] deltaActual = new float[h1.Length];
        float[] deltaRef = new float[h1.Length];
        for (int i = 0; i < deltaActual.Length; i++)
        {
            deltaActual[i] = h2[i] - h1[i];
            deltaRef[i] = refHidden[i] - refCrossInput[i];
        }
        Report("cross layer delta", deltaActual, deltaRef, out double deltaRelative);
        Assert.True(deltaRelative < RelativeTolerance, $"Cross-layer delta relative {deltaRelative:G6} >= {RelativeTolerance}.");
    }

    private static void Report(string label, ReadOnlySpan<float> actual, ReadOnlySpan<float> reference, out double relative)
    {
        double maxDiff = 0, maxRef = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            maxDiff = Math.Max(maxDiff, Math.Abs(actual[i] - reference[i]));
            maxRef = Math.Max(maxRef, Math.Abs(reference[i]));
        }
        relative = maxDiff / maxRef;
        Console.WriteLine($"MOSS-VL text parity, {label}: max|diff| {maxDiff:G6}, max|ref| {maxRef:G6}, relative {relative:G6}.");
    }
}
