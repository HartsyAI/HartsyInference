using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Models.MossVl;

namespace HartsyInference.LLM.Tests;

/// <summary>Prefill and one KV-cached decode step of <see cref="MossVlDecodeSession"/> on the real checkpoint's first three layers,
/// against the numpy transcription in <c>tests/python-reference/dump_moss_vl_text.py</c>. The reference's decode logits come from a full
/// recompute over the prompt plus the greedy token, so they also check the cache and the decode token's position.</summary>
[Trait("Category", "Integration")]
public sealed class MossVlGenerationParityTests
{
    private const string CheckpointVariable = "HARTSY_MOSS_VL_INDEX";
    private const string VisionVariable = "HARTSY_MOSS_VL_REF";
    private const string TextVariable = "HARTSY_MOSS_VL_TEXT_REF";
    private const int Layers = 3;
    private const int MergedHeight = 8, MergedWidth = 12;
    private const double RelativeTolerance = 1e-2;

    [RealWeightsFact(CheckpointVariable, VisionVariable, TextVariable)]
    public void PrefillAndDecodeLogitsMatchTheReference()
    {
        string checkpoint = Path.GetDirectoryName(Environment.GetEnvironmentVariable(CheckpointVariable)!)!;
        string mergedPath = Environment.GetEnvironmentVariable(VisionVariable)!;
        string refDir = Path.GetDirectoryName(Environment.GetEnvironmentVariable(TextVariable)!)!;
        int[] ids = File.ReadAllText(Path.Combine(refDir, "prompt_ids.txt"))
            .Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();
        int stepToken = int.Parse(File.ReadAllText(Path.Combine(refDir, "step_token.txt")).Trim());

        MossVlConfig config;
        using (JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(checkpoint, "config.json"))))
            config = MossVlConfig.FromJson(doc.RootElement);

        (long[] mergedShape, float[] mergedData) = NpyReader.ReadFloat32(mergedPath);
        (_, float[] refPrefill) = NpyReader.ReadFloat32(Path.Combine(refDir, "prefill_logits.npy"));
        (_, float[] refStep) = NpyReader.ReadFloat32(Path.Combine(refDir, "step_logits.npy"));

        using MossVlTextDecoder decoder = MossVlTextDecoder.Open(checkpoint, config, Layers);
        using IBackend backend = new CpuBackend();
        using Tensor merged = new(new TensorShape(mergedShape[0], mergedShape[1]), DType.F32);
        mergedData.AsSpan().CopyTo(merged.AsSpan<float>());

        using MossVlDecodeSession session = new(decoder, config, backend, ids, merged, MergedHeight, MergedWidth, maxNewTokens: 1);
        float[] prefill = session.Prefill();
        float[] step = session.Step(stepToken);

        Assert.Equal(Argmax(refPrefill), Argmax(prefill));
        double prefillRelative = Relative(prefill, refPrefill);
        double stepRelative = Relative(step, refStep);
        Console.WriteLine($"MOSS-VL logits ({Layers} layers): prefill relative {prefillRelative:G6}, decode relative {stepRelative:G6}.");
        Assert.True(prefillRelative < RelativeTolerance, $"Prefill logits relative {prefillRelative:G6} >= {RelativeTolerance}.");
        Assert.True(stepRelative < RelativeTolerance, $"Decode logits relative {stepRelative:G6} >= {RelativeTolerance}.");
        Assert.Equal(Argmax(refStep), Argmax(step));
    }

    /// <summary>The largest absolute difference over the largest absolute reference value.</summary>
    private static double Relative(float[] actual, float[] reference)
    {
        Assert.Equal(reference.Length, actual.Length);
        double maxDiff = 0, maxRef = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            maxDiff = Math.Max(maxDiff, Math.Abs(actual[i] - reference[i]));
            maxRef = Math.Max(maxRef, Math.Abs(reference[i]));
        }
        return maxDiff / maxRef;
    }

    private static int Argmax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }
}
