using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Pins the fold-first contract on every image converter that now takes its input from
/// <c>CheckpointSource</c>. A converter renames and splits <c>.weight</c> but has no rule for <c>.weight_scale</c>, so
/// a companion still present when it runs is dropped and the weight is served as <c>real/scale</c> — no symptom at
/// load, noise at the end of the generation. Each converter must refuse that dictionary, and must accept the folded
/// one unchanged.</summary>
public sealed class ImageConverterFoldedCompanionTests
{
    private static Tensor Weight(int rows, int columns) => new Tensor(new TensorShape(rows, columns), DType.BF16);

    private static Tensor Scale()
    {
        Tensor scale = new Tensor(new TensorShape(1), DType.F32);
        scale.AsSpan<float>()[0] = 0.0125f;
        return scale;
    }

    private static void DisposeAll(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor tensor in weights.Values) tensor.Dispose();
    }

    private static void AssertRefusesUnfolded(string companionKey, Action<Dictionary<string, Tensor>> convert,
        Dictionary<string, Tensor> folded)
    {
        Dictionary<string, Tensor> unfolded = new(folded) { [companionKey] = Scale() };
        try
        {
            NotSupportedException error = Assert.Throws<NotSupportedException>(() => convert(unfolded));
            Assert.Contains(companionKey, error.Message, StringComparison.Ordinal);
            Assert.Contains("CheckpointSource.Open", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            unfolded[companionKey].Dispose();
        }
    }

    [Fact]
    public void ChromaCheckpointConverter_RefusesAnUnfoldedDictionaryAndAcceptsAFoldedOne()
    {
        Dictionary<string, Tensor> folded = new() { ["img_in.weight"] = Weight(3072, 64) };
        try
        {
            AssertRefusesUnfolded("img_in.weight_scale", w => ChromaCheckpointConverter.Convert(w), folded);

            ChromaCheckpointConverter.ConvertedWeights converted = ChromaCheckpointConverter.Convert(folded);
            Assert.Same(folded["img_in.weight"], converted.Transformer["x_embedder.weight"]);
        }
        finally
        {
            DisposeAll(folded);
        }
    }

    [Fact]
    public void ZImageCheckpointConverter_RefusesAnUnfoldedDictionaryAndAcceptsAFoldedOne()
    {
        Dictionary<string, Tensor> folded = new() { ["x_embedder.weight"] = Weight(3840, 64) };
        try
        {
            AssertRefusesUnfolded("x_embedder.weight_scale", w => ZImageCheckpointConverter.Convert(w), folded);

            ZImageCheckpointConverter.ConvertedWeights converted = ZImageCheckpointConverter.Convert(folded);
            Assert.Same(folded["x_embedder.weight"], converted.Transformer["x_embedder.weight"]);
        }
        finally
        {
            DisposeAll(folded);
        }
    }

    [Fact]
    public void ZetaChromaCheckpointConverter_RefusesAnUnfoldedDictionaryAndAcceptsAFoldedOne()
    {
        Dictionary<string, Tensor> folded = new() { ["dec_net.0.weight"] = Weight(64, 64) };
        try
        {
            AssertRefusesUnfolded("dec_net.0.weight_scale", w => ZetaChromaCheckpointConverter.Convert(w), folded);

            ZImageCheckpointConverter.ConvertedWeights converted = ZetaChromaCheckpointConverter.Convert(folded);
            Assert.Same(folded["dec_net.0.weight"], converted.Transformer["dec_net.0.weight"]);
        }
        finally
        {
            DisposeAll(folded);
        }
    }

    [Fact]
    public void Lumina2CheckpointConverter_RefusesAnUnfoldedDictionaryAndAcceptsAFoldedOne()
    {
        Dictionary<string, Tensor> folded = new() { ["x_embedder.weight"] = Weight(2304, 64) };
        try
        {
            AssertRefusesUnfolded("x_embedder.weight_scale", w => Lumina2CheckpointConverter.Convert(w), folded);

            Lumina2CheckpointConverter.ConvertedWeights converted = Lumina2CheckpointConverter.Convert(folded);
            Assert.Same(folded["x_embedder.weight"], converted.Transformer["x_embedder.weight"]);
        }
        finally
        {
            DisposeAll(folded);
        }
    }

    [Fact]
    public void AuraFlowCheckpointConverter_RefusesAnUnfoldedDictionaryAndAcceptsAFoldedOne()
    {
        Dictionary<string, Tensor> folded = new() { ["init_x_linear.weight"] = Weight(3072, 64) };
        try
        {
            AssertRefusesUnfolded("init_x_linear.weight_scale", w => AuraFlowCheckpointConverter.Convert(w), folded);

            AuraFlowCheckpointConverter.ConvertedWeights converted = AuraFlowCheckpointConverter.Convert(folded);
            Assert.Same(folded["init_x_linear.weight"], converted.Transformer["pos_embed.proj.weight"]);
        }
        finally
        {
            DisposeAll(folded);
        }
    }

    [Fact]
    public void AnimaCheckpointConverter_RefusesAnUnfoldedDictionaryAndAcceptsAFoldedOne()
    {
        Dictionary<string, Tensor> folded = new() { ["net.x_embedder.proj.1.weight"] = Weight(2048, 64) };
        try
        {
            AssertRefusesUnfolded("net.x_embedder.proj.1.weight_scale", w => AnimaCheckpointConverter.Convert(w), folded);

            AnimaCheckpointConverter.ConvertedWeights converted = AnimaCheckpointConverter.Convert(folded);
            Assert.Same(folded["net.x_embedder.proj.1.weight"], converted.Transformer["x_embedder.proj.1.weight"]);
        }
        finally
        {
            DisposeAll(folded);
        }
    }

    [Fact]
    public void HiDreamCheckpointConverter_RefusesAnUnfoldedDictionaryAndAcceptsAFoldedOne()
    {
        Dictionary<string, Tensor> folded = new() { ["double_stream_blocks.0.attn.to_q.weight"] = Weight(2560, 64) };
        try
        {
            AssertRefusesUnfolded("double_stream_blocks.0.attn.to_q.scale_weight",
                w => HiDreamCheckpointConverter.Convert(w), folded);

            HiDreamCheckpointConverter.ConvertedWeights converted = HiDreamCheckpointConverter.Convert(folded);
            Assert.Same(folded["double_stream_blocks.0.attn.to_q.weight"],
                converted.Transformer["double_stream_blocks.0.attn.to_q.weight"]);
        }
        finally
        {
            DisposeAll(folded);
        }
    }

    [Fact]
    public void OmniGen2CheckpointConverter_RefusesAnUnfoldedDictionaryAndAcceptsAFoldedOne()
    {
        Dictionary<string, Tensor> folded = new() { ["transformer.x_embedder.weight"] = Weight(2520, 64) };
        try
        {
            AssertRefusesUnfolded("transformer.x_embedder.weight_scale", w => OmniGen2CheckpointConverter.Convert(w), folded);

            OmniGen2CheckpointConverter.ConvertedWeights converted = OmniGen2CheckpointConverter.Convert(folded);
            Assert.Same(folded["transformer.x_embedder.weight"], converted.Transformer["x_embedder.weight"]);
        }
        finally
        {
            DisposeAll(folded);
        }
    }

    [Fact]
    public void Kandinsky5CheckpointConverter_RefusesAnUnfoldedDictionaryAndAcceptsAFoldedOne()
    {
        Dictionary<string, Tensor> folded = new() { ["visual_embeddings.in_layer.weight"] = Weight(1792, 64) };
        try
        {
            AssertRefusesUnfolded("visual_embeddings.in_layer.weight_scale",
                w => Kandinsky5CheckpointConverter.Convert(w), folded);

            Kandinsky5CheckpointConverter.ConvertedWeights converted = Kandinsky5CheckpointConverter.Convert(folded);
            Assert.Same(folded["visual_embeddings.in_layer.weight"],
                converted.Transformer["visual_embeddings.in_layer.weight"]);
        }
        finally
        {
            DisposeAll(folded);
        }
    }

    [Fact]
    public void LanceCheckpointConverter_RefusesAnUnfoldedDictionaryAndAcceptsAFoldedOne()
    {
        // The backbone prefix is what makes this converter fold-order sensitive: it strips
        // `language_model.model.` from the weight, and a companion folded afterwards would pair nothing.
        Dictionary<string, Tensor> folded = new() { ["language_model.model.layers.0.mlp.up_proj.weight"] = Weight(2048, 64) };
        try
        {
            AssertRefusesUnfolded("language_model.model.layers.0.mlp.up_proj.weight_scale",
                w => LanceCheckpointConverter.Convert(w), folded);

            LanceCheckpointConverter.ConvertedWeights converted = LanceCheckpointConverter.Convert(folded);
            Assert.Same(folded["language_model.model.layers.0.mlp.up_proj.weight"],
                converted.Transformer["layers.0.mlp.up_proj.weight"]);
        }
        finally
        {
            DisposeAll(folded);
        }
    }

    [Fact]
    public void ChromaCheckpointConverter_SplitsABlockQuantFusedQkvByteForByte()
    {
        // The split used to size its copies from DType.SizeInBytes, which is 0 for every block quant: on a GGUF it
        // produced three correctly-shaped all-zero projections while the dense path stayed byte-perfect.
        const int innerDim = 3072;
        const int inDim = 64;
        Tensor fused = new Tensor(new TensorShape(3 * innerDim, inDim), DType.Q8_0);
        Span<byte> bytes = fused.AsSpan<byte>();
        for (int i = 0; i < bytes.Length; i++) bytes[i] = (byte)(i % 251 + 1);
        Dictionary<string, Tensor> folded = new() { ["double_blocks.0.img_attn.qkv.weight"] = fused };
        ChromaCheckpointConverter.ConvertedWeights converted = ChromaCheckpointConverter.Convert(folded);
        try
        {
            ReadOnlySpan<byte> source = fused.AsReadOnlySpan<byte>();
            int chunkBytes = source.Length / 3;
            int offset = 0;
            foreach (string name in new[] { "to_q", "to_k", "to_v" })
            {
                Tensor split = converted.Transformer[$"transformer_blocks.0.attn.{name}.weight"];
                Assert.Equal(DType.Q8_0, split.DType);
                Assert.Equal(innerDim, (int)split.Shape[0]);
                Assert.Equal(inDim, (int)split.Shape[1]);
                Assert.True(split.AsReadOnlySpan<byte>().SequenceEqual(source.Slice(offset, chunkBytes)));
                offset += chunkBytes;
            }
        }
        finally
        {
            foreach (Tensor tensor in converted.Transformer.Values) tensor.Dispose();
            fused.Dispose();
        }
    }

    [Fact]
    public void ZetaChromaCheckpointConverter_RefusesToFuseSplitAttentionThatCarriesPerRowScales()
    {
        // Fusing Q/K/V along dim 0 concatenates rows, and int8_tensorwise scales are indexed by row. Keeping only
        // Q's would run K and V at another projection's magnitude, which renders rather than fails.
        Dictionary<string, Tensor> weights = new()
        {
            ["layers.0.attention.to_q.weight"] = new Tensor(new TensorShape(4, 256), DType.I8),
            ["layers.0.attention.to_k.weight"] = new Tensor(new TensorShape(4, 256), DType.I8),
            ["layers.0.attention.to_v.weight"] = new Tensor(new TensorShape(4, 256), DType.I8),
        };
        using Tensor rowScale = new Tensor(new TensorShape(4, 1), DType.F32);
        weights["layers.0.attention.to_q.weight"].QuantInfo =
            new QuantWeightInfo { Format = "int8_tensorwise", RowScale = rowScale, ConvRotGroupSize = 256 };
        try
        {
            NotSupportedException error = Assert.Throws<NotSupportedException>(
                () => ZetaChromaCheckpointConverter.Convert(weights));
            Assert.Contains("int8_tensorwise", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void ZetaChromaCheckpointConverter_RefusesToFuseSplitAttentionStoredInThreeDtypes()
    {
        // The fused tensor is allocated entirely in Q's dtype, so a cheaper K leaves it under-filled, and a cheaper
        // Q lets the copies run past it.
        Dictionary<string, Tensor> weights = new()
        {
            ["layers.0.attention.to_q.weight"] = new Tensor(new TensorShape(4, 256), DType.Q8_0),
            ["layers.0.attention.to_k.weight"] = new Tensor(new TensorShape(4, 256), DType.Q4_0),
            ["layers.0.attention.to_v.weight"] = new Tensor(new TensorShape(4, 256), DType.Q8_0),
        };
        try
        {
            NotSupportedException error = Assert.Throws<NotSupportedException>(
                () => ZetaChromaCheckpointConverter.Convert(weights));
            Assert.Contains("Q8_0/Q4_0/Q8_0", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            DisposeAll(weights);
        }
    }
}
