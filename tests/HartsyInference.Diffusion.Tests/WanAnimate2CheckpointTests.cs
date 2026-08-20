using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-checkpoint conversion gate for Wan-Animate-2. The failure mode this pins is a converter that
/// reports "converted N keys" while silently discarding some: every assert here is about the PARTITION of the
/// file's 2263 tensors being exhaustive, and about all 480 int8 Linears carrying both a per-row scale and a ConvRot
/// descriptor (an int8 weight consumed without either produces plausible-looking garbage, never an error).</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class WanAnimate2CheckpointTests
{
    private readonly ITestOutputHelper _output;
    public WanAnimate2CheckpointTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Animate2Checkpoint_ConvertsExhaustively_WithInt8ConvRotAttachedToEveryQuantizedLinear()
    {
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.WanVideo.Animate2)) return;

        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(TestPaths.WanVideo.Animate2);
        Dictionary<string, Tensor> raw = loader.GetAllTensors();
        _output.WriteLine($"file tensors: {raw.Count}");
        Assert.Equal(2263, raw.Count);

        HashSet<string> rawKeys = new HashSet<string>(raw.Keys, StringComparer.Ordinal);
        int rawInt8 = raw.Count(kvp => kvp.Value.DType == DType.I8);

        Assert.True(WanVideoCheckpointConverter.IsAnimate2Metadata(loader.Metadata),
            "__metadata__ did not declare model_type 'animate2'.");
        WanVideoCheckpointConverter.ConvertedWeights converted =
            WanVideoCheckpointConverter.Convert(raw, loader.Metadata);
        Assert.True(converted.IsAnimate2);
        Dictionary<string, Tensor> w = converted.Transformer;

        // Exhaustive partition: every source key is either consumed (renamed into the transformer bucket) or is one
        // of the two companion suffixes AttachInt8QuantInfo folds onto QuantInfo. Nothing else may vanish.
        List<string> consumedSources = [];
        List<string> companions = [];
        List<string> unaccounted = [];
        foreach (string key in rawKeys)
        {
            if (key.EndsWith(".weight_scale", StringComparison.Ordinal) || key.EndsWith(".comfy_quant", StringComparison.Ordinal))
            {
                companions.Add(key);
                continue;
            }
            string? mapped = WanVideoCheckpointConverter.MapKey(key, fromOriginalNaming: true);
            if (mapped is not null && w.ContainsKey(mapped)) consumedSources.Add(key);
            else unaccounted.Add(key);
        }
        _output.WriteLine($"consumed: {consumedSources.Count}  companions: {companions.Count}  unaccounted: {unaccounted.Count}");
        Assert.Empty(unaccounted);
        Assert.Equal(480, rawInt8);
        Assert.Equal(960, companions.Count);
        Assert.Equal(raw.Count, consumedSources.Count + companions.Count);
        // Renames must not collide: a two-to-one mapping would drop a tensor without either count noticing.
        Assert.Equal(consumedSources.Count, w.Count);

        WanVideoConfig config = WanConfigDetector.Detect(w, converted.IsAnimate2);
        _output.WriteLine(WanConfigDetector.Describe(config));
        Assert.Equal(40, config.NumLayers);
        Assert.Equal(5120, config.InnerDim);
        Assert.Equal(40, config.NumHeads);
        Assert.Equal(13824, config.FfnDim);
        Assert.Equal(36, config.InChannels);
        Assert.Equal(16, config.OutChannels);
        Assert.Equal(1280, config.ImageDim);
        Assert.True(config.IsAnimate2);
        Assert.False(config.IsAnimate);

        // The converted key set must be exactly what a Wan i2v DiT needs — no extras, no gaps.
        HashSet<string> expected = ExpectedI2vKeys(config.NumLayers);
        HashSet<string> actual = new HashSet<string>(w.Keys, StringComparer.Ordinal);
        _output.WriteLine($"missing: {string.Join(", ", expected.Except(actual).Order())}");
        _output.WriteLine($"extra:   {string.Join(", ", actual.Except(expected).Order())}");
        Assert.Equal(expected.OrderBy(k => k, StringComparer.Ordinal), actual.OrderBy(k => k, StringComparer.Ordinal));

        // Per-row scale orientation: RowScale indexes weight rows (dim 0) — CudaBackend.EnsureInt8RowScaleDev reads
        // n = weight.Shape[0] entries. The 80 non-square ffn layers are what make this checkable at all: ffn.0 is
        // [13824, 5120] and ffn.2 is [5120, 13824], so a transposed scale is the wrong length on both.
        int quantized = 0, convRot = 0, nonSquare = 0;
        foreach ((string key, Tensor t) in w)
        {
            if (t.DType != DType.I8) continue;
            quantized++;
            QuantWeightInfo info = Assert.IsType<QuantWeightInfo>(t.QuantInfo);
            Tensor scale = Assert.IsType<Tensor>(info.RowScale);
            Assert.Equal(DType.F32, scale.DType);
            Assert.Equal(t.Shape[0], scale.ElementCount);
            Assert.Equal(256, info.ConvRotGroupSize);
            Assert.False(info.FullPrecisionMatMul);
            convRot++;
            if (t.Shape[0] != t.Shape[1]) nonSquare++;
            _ = key;
        }
        _output.WriteLine($"int8 linears: {quantized} (convrot {convRot}, non-square {nonSquare})");
        Assert.Equal(480, quantized);
        Assert.Equal(480, convRot);
        Assert.Equal(80, nonSquare);
    }

    /// <summary>Every diffusers-named key a Wan2.1 I2V-14B DiT carries, by name only — <see cref="WanSyntheticWeights"/>
    /// would allocate the 14B parameters behind them.</summary>
    private static HashSet<string> ExpectedI2vKeys(int layers)
    {
        HashSet<string> keys = new HashSet<string>(StringComparer.Ordinal)
        {
            "patch_embedding.weight", "patch_embedding.bias",
            "proj_out.weight", "proj_out.bias", "scale_shift_table",
            "condition_embedder.time_embedder.linear_1.weight", "condition_embedder.time_embedder.linear_1.bias",
            "condition_embedder.time_embedder.linear_2.weight", "condition_embedder.time_embedder.linear_2.bias",
            "condition_embedder.time_proj.weight", "condition_embedder.time_proj.bias",
            "condition_embedder.text_embedder.linear_1.weight", "condition_embedder.text_embedder.linear_1.bias",
            "condition_embedder.text_embedder.linear_2.weight", "condition_embedder.text_embedder.linear_2.bias",
            "condition_embedder.image_embedder.norm1.weight", "condition_embedder.image_embedder.norm1.bias",
            "condition_embedder.image_embedder.ff.net.0.proj.weight", "condition_embedder.image_embedder.ff.net.0.proj.bias",
            "condition_embedder.image_embedder.ff.net.2.weight", "condition_embedder.image_embedder.ff.net.2.bias",
            "condition_embedder.image_embedder.norm2.weight", "condition_embedder.image_embedder.norm2.bias",
        };
        for (int i = 0; i < layers; i++)
        {
            string p = $"blocks.{i}";
            keys.Add($"{p}.scale_shift_table");
            keys.Add($"{p}.norm2.weight"); keys.Add($"{p}.norm2.bias");
            foreach (string a in new[] { "attn1", "attn2" })
                foreach (string proj in new[] { "to_q", "to_k", "to_v", "to_out.0" })
                {
                    keys.Add($"{p}.{a}.{proj}.weight"); keys.Add($"{p}.{a}.{proj}.bias");
                }
            foreach (string a in new[] { "attn1", "attn2" })
            {
                keys.Add($"{p}.{a}.norm_q.weight"); keys.Add($"{p}.{a}.norm_k.weight");
            }
            keys.Add($"{p}.attn2.add_k_proj.weight"); keys.Add($"{p}.attn2.add_k_proj.bias");
            keys.Add($"{p}.attn2.add_v_proj.weight"); keys.Add($"{p}.attn2.add_v_proj.bias");
            keys.Add($"{p}.attn2.norm_added_k.weight");
            keys.Add($"{p}.ffn.net.0.proj.weight"); keys.Add($"{p}.ffn.net.0.proj.bias");
            keys.Add($"{p}.ffn.net.2.weight"); keys.Add($"{p}.ffn.net.2.bias");
        }
        return keys;
    }
}
