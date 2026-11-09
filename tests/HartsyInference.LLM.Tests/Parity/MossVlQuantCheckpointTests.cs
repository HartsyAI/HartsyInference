using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.LLM.Tests;

/// <summary>The official MOSS-VL NF4 (bitsandbytes double-quant) and FP8 (compressed-tensors, per-channel) releases read against the BF16
/// checkpoint they were made from: every quantized projection is indexed with the storage its config declares, and decoded weights
/// match the BF16 ones within each format's error. Header and per-tensor reads only; no model is built.
/// <c>HARTSY_MOSS_VL_INDEX</c>, <c>HARTSY_MOSS_VL_NF4_INDEX</c> and <c>HARTSY_MOSS_VL_FP8_INDEX</c> are the three
/// <c>model.safetensors.index.json</c> files.</summary>
[Trait("Category", "Integration")]
public sealed class MossVlQuantCheckpointTests
{
    private const string Bf16Variable = "HARTSY_MOSS_VL_INDEX";
    private const string Nf4Variable = "HARTSY_MOSS_VL_NF4_INDEX";
    private const string Fp8Variable = "HARTSY_MOSS_VL_FP8_INDEX";

    private static string Dir(string variable) => Path.GetDirectoryName(Environment.GetEnvironmentVariable(variable)!)!;

    /// <summary>Relative RMS error of <paramref name="actual"/> against <paramref name="reference"/>.</summary>
    private static double RelativeRms(ReadOnlySpan<float> actual, ReadOnlySpan<float> reference)
    {
        double err = 0, norm = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double d = actual[i] - reference[i];
            err += d * d;
            norm += (double)reference[i] * reference[i];
        }
        return Math.Sqrt(err / norm);
    }

    [RealWeightsFact(Nf4Variable)]
    public void Nf4IndexesEveryQuantizedProjectionAndKeepsTheSkippedModulesPlain()
    {
        using HfQuantCheckpoint nf4 = HfQuantCheckpoint.Open(Dir(Nf4Variable));
        Dictionary<HfWeightStorage, int> counts = nf4.Weights.Values.GroupBy(w => w.Storage).ToDictionary(g => g.Key, g => g.Count());
        Console.WriteLine($"NF4 storage: {string.Join(", ", counts.Select(kv => $"{kv.Key} {kv.Value}"))}");
        // Layers 4–43 are quantized except cross_attn: 30 self-attention layers x 7 projections + 10 cross layers x 3 MLP projections.
        Assert.Equal(240, counts[HfWeightStorage.Nf4]);
        Assert.Equal(HfWeightStorage.Plain, nf4.Weights["model.language_model.layers.0.self_attn.q_proj.weight"].Storage);
        Assert.Equal(HfWeightStorage.Plain, nf4.Weights["model.language_model.layers.6.cross_attn.q_proj.weight"].Storage);
        Assert.Equal(HfWeightStorage.Nf4, nf4.Weights["model.language_model.layers.6.mlp.up_proj.weight"].Storage);
        Assert.Equal(HfWeightStorage.Plain, nf4.Weights["lm_head.weight"].Storage);
    }

    [RealWeightsFact(Fp8Variable)]
    public void Fp8IndexesThePerChannelProjectionsOfTheSelfAttentionLayers()
    {
        using HfQuantCheckpoint fp8 = HfQuantCheckpoint.Open(Dir(Fp8Variable));
        Dictionary<HfWeightStorage, int> counts = fp8.Weights.Values.GroupBy(w => w.Storage).ToDictionary(g => g.Key, g => g.Count());
        Console.WriteLine($"FP8 storage: {string.Join(", ", counts.Select(kv => $"{kv.Key} {kv.Value}"))}");
        // 36 self-attention layers x 7 projections; the 12 cross layers stay BF16.
        Assert.Equal(252, counts[HfWeightStorage.Fp8Channel]);
        Assert.DoesNotContain(fp8.Weights.Keys, k => k.EndsWith("weight_scale", StringComparison.Ordinal));
        Assert.Equal(HfWeightStorage.Plain, fp8.Weights["model.language_model.layers.6.mlp.up_proj.weight"].Storage);
    }

    [RealWeightsFact(Bf16Variable, Nf4Variable, Fp8Variable)]
    public void DecodedWeightsMatchTheBf16CheckpointWithinEachFormatsError()
    {
        using HfQuantCheckpoint bf16 = HfQuantCheckpoint.Open(Dir(Bf16Variable));
        using HfQuantCheckpoint nf4 = HfQuantCheckpoint.Open(Dir(Nf4Variable));
        using HfQuantCheckpoint fp8 = HfQuantCheckpoint.Open(Dir(Fp8Variable));
        foreach (string name in (string[])["model.language_model.layers.20.self_attn.q_proj.weight", "model.language_model.layers.33.mlp.down_proj.weight"])
        {
            using Tensor reference = bf16.DequantizeF32(name);
            using Tensor fromNf4 = nf4.DequantizeF32(name);
            using Tensor fromFp8 = fp8.DequantizeF32(name);
            Assert.Equal(reference.Shape, fromNf4.Shape);
            Assert.Equal(reference.Shape, fromFp8.Shape);
            double nf4Error = RelativeRms(fromNf4.AsReadOnlySpan<float>(), reference.AsReadOnlySpan<float>());
            double fp8Error = RelativeRms(fromFp8.AsReadOnlySpan<float>(), reference.AsReadOnlySpan<float>());
            using Tensor q4 = nf4.Produce(name, DType.Q4_0);
            using Tensor q4F32 = ModelAssets.Gguf.GgufDequantizer.Dequantize(q4, DType.F32);
            double q4Error = RelativeRms(q4F32.AsReadOnlySpan<float>(), reference.AsReadOnlySpan<float>());
            using Tensor q4k = nf4.Produce(name, DType.Q4_K);
            using Tensor q4kF32 = ModelAssets.Gguf.GgufDequantizer.Dequantize(q4k, DType.F32);
            double q4kError = RelativeRms(q4kF32.AsReadOnlySpan<float>(), reference.AsReadOnlySpan<float>());
            using Tensor q8 = fp8.Produce(name, DType.Q8_0);
            using Tensor q8F32 = ModelAssets.Gguf.GgufDequantizer.Dequantize(q8, DType.F32);
            double q8Error = RelativeRms(q8F32.AsReadOnlySpan<float>(), reference.AsReadOnlySpan<float>());
            Console.WriteLine($"{name}: NF4 rel RMS {nf4Error:F4}, NF4→Q4_0 {q4Error:F4}, NF4→Q4_K {q4kError:F4}, FP8 {fp8Error:F4}, FP8→Q8_0 {q8Error:F4}");
            Assert.InRange(q4kError, 0, 0.20);
            Assert.InRange(q8Error, 0, 0.05);
            // NF4 at block 64 sits near 10% relative RMS; the Q4_0 repack adds its own rounding; FP8 E4M3 per channel near 3%.
            Assert.InRange(nf4Error, 0, 0.15);
            Assert.InRange(q4Error, 0, 0.20);
            Assert.InRange(fp8Error, 0, 0.05);
        }
    }
}
