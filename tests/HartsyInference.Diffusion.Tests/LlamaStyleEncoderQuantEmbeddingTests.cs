using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>An <c>int8_tensorwise</c> token embedding table has to be dequantized, not widened.
/// <see cref="DType.I8"/> reports <c>IsQuantized == false</c>, so <c>Tensor.CastTo</c> accepts it and returns the
/// raw sbyte values as floats — every embedding row ~100× its true magnitude, with the ConvRot rotation still
/// baked in. That is a wrong image, not a load error. Comfy-Org's Qwen-Image 2.1 encoder
/// (<c>qwen3vl_8b_int8_convrot</c>) quantizes this table; their LTX-2.5 Gemma-4 one leaves it BF16, which is why
/// no shipped encoder had reached the path before.</summary>
public sealed class LlamaStyleEncoderQuantEmbeddingTests
{
    // ConvRot only rotates when in_features is a multiple of the group size, so hidden must be 256 here.
    private const int Hidden = 256;
    private const int Vocab = 4;

    private static LlamaStyleEncoderConfig ZeroLayerConfig => new()
    {
        HiddenSize = Hidden,
        NumLayers = 0,
        NumQueryHeads = 1,
        NumKvHeads = 1,
        HeadDim = Hidden,
        IntermediateSize = Hidden,
        VocabSize = Vocab,
        HasFinalNorm = false,
    };

    [Fact]
    public void AnInt8ConvRotEmbeddingIsDequantizedRatherThanWidened()
    {
        Tensor original = BuildTable();
        float[] expected = original.AsReadOnlySpan<float>().ToArray();

        // QuantizeFromF32 consumes its input (the rotation runs in place), so it takes the copy.
        Tensor toQuantize = BuildTable();
        (Tensor packed, Tensor rowScale) = Int8ConvRotCodec.QuantizeFromF32(toQuantize, convRotGroupSize: 256);
        packed.QuantInfo = new QuantWeightInfo
        {
            Format = "int8_tensorwise",
            RowScale = rowScale,
            ConvRotGroupSize = 256,
        };
        Assert.Equal(DType.I8, packed.DType);

        using LlamaStyleEncoder encoder = new LlamaStyleEncoder(ZeroLayerConfig);
        encoder.LoadWeights(new Dictionary<string, Tensor> { ["model.embed_tokens.weight"] = packed });

        using Tensor looked = encoder.LookupEmbeddings([0, 1, 2, 3]);
        ReadOnlySpan<float> actual = looked.AsReadOnlySpan<float>();
        Assert.Equal(expected.Length, actual.Length);

        // int8 with a per-row absmax/127 scale: worst case is half a step of that row's own scale.
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
        }
        Assert.True(worst < 0.01f, $"dequantized embedding is {worst} off the source table; int8 at this scale should be under 0.01.");

        // The negative control. The bug this guards is not "slightly off" — a raw widen leaves values in the
        // sbyte range, so the table's own magnitude is the thing that distinguishes the two outcomes.
        float maxMagnitude = 0f;
        foreach (float value in actual)
        {
            maxMagnitude = MathF.Max(maxMagnitude, MathF.Abs(value));
        }
        Assert.True(maxMagnitude < 2.0f, $"embedding magnitude {maxMagnitude} is in the raw-sbyte range — the scale was dropped.");

        packed.Dispose();
        rowScale.Dispose();
        original.Dispose();
    }

    /// <summary>A plain BF16 table still loads unchanged — the int8 branch must not capture the ordinary case.</summary>
    [Fact]
    public void ABf16EmbeddingIsStillWidenedDirectly()
    {
        using Tensor f32 = BuildTable();
        Tensor bf16 = f32.CastTo(DType.BF16);

        using LlamaStyleEncoder encoder = new LlamaStyleEncoder(ZeroLayerConfig);
        encoder.LoadWeights(new Dictionary<string, Tensor> { ["model.embed_tokens.weight"] = bf16 });

        using Tensor looked = encoder.LookupEmbeddings([1]);
        ReadOnlySpan<float> actual = looked.AsReadOnlySpan<float>();
        ReadOnlySpan<float> expected = f32.AsReadOnlySpan<float>();
        for (int i = 0; i < Hidden; i++)
        {
            Assert.Equal(expected[Hidden + i], actual[i], 2);
        }
        bf16.Dispose();
    }

    /// <summary>Values spread over a realistic embedding range, different per row so a per-row scale matters.</summary>
    private static Tensor BuildTable()
    {
        Tensor table = new Tensor(new TensorShape(Vocab, Hidden), DType.F32);
        Span<float> span = table.AsSpan<float>();
        Random rng = new Random(9);
        for (int row = 0; row < Vocab; row++)
        {
            float rowScale = 0.02f * (row + 1);
            for (int column = 0; column < Hidden; column++)
            {
                span[row * Hidden + column] = (float)(rng.NextDouble() * 2.0 - 1.0) * rowScale;
            }
        }
        return table;
    }
}
