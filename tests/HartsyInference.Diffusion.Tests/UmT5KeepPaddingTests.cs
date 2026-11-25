using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The umT5 batch encode on a tiny random encoder (no pretrained weights): the padded-keeping path returns every row the encoder produced,
/// and against the zeroing path it differs only on the padded rows, where the zeroing path writes zeros. It also refuses a token set whose length is
/// not the batch's.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class UmT5KeepPaddingTests
{
    private const int DModel = 16, Heads = 2, DKv = 8, DFf = 32, Layers = 1, Vocab = 20, Buckets = 32, Length = 4;

    // Token 0 is the padding id. The second context has two real tokens and two pad positions.
    private static readonly int[] Full = [5, 6, 7, 1];
    private static readonly int[] Padded = [4, 1, 0, 0];

    [Fact]
    public void KeepingPaddingDiffersFromZeroingOnlyOnThePaddedRows()
    {
        Dictionary<string, Tensor> weights = TinyWeights(seed: 61);
        try
        {
            using T5TextEncoder encoder = Encoder(weights);
            using CpuBackend backend = new();

            Tensor[] kept = VideoRecipeUtils.EncodeUmT5PromptBatchKeepingPadding(backend, encoder, DModel, Length, Full, Padded);
            Tensor[] zeroed = VideoRecipeUtils.EncodeUmT5PromptBatch(backend, encoder, DModel, Length, zeroPaddedRows: true, [Full, Padded]);
            try
            {
                for (int context = 0; context < 2; context++)
                {
                    float[] keptRows = kept[context].AsReadOnlySpan<float>().ToArray();
                    float[] zeroedRows = zeroed[context].AsReadOnlySpan<float>().ToArray();
                    Assert.Equal(Length * DModel, keptRows.Length);
                    for (int row = 0; row < Length; row++)
                    {
                        bool isPad = context == 1 && row >= 2;
                        for (int c = 0; c < DModel; c++)
                        {
                            int at = row * DModel + c;
                            if (isPad)
                            {
                                Assert.NotEqual(0f, keptRows[at]);
                                Assert.Equal(0f, zeroedRows[at]);
                            }
                            else
                            {
                                Assert.Equal(zeroedRows[at], keptRows[at]);
                            }
                        }
                    }
                }
                Assert.All(kept.SelectMany(t => t.AsReadOnlySpan<float>().ToArray()), v => Assert.True(float.IsFinite(v)));
            }
            finally
            {
                foreach (Tensor t in kept.Concat(zeroed))
                    t.Dispose();
            }
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    [Fact]
    public void ATokenSetOfTheWrongLengthIsRefused()
    {
        Dictionary<string, Tensor> weights = TinyWeights(seed: 62);
        try
        {
            using T5TextEncoder encoder = Encoder(weights);
            using CpuBackend backend = new();
            int[] tooShort = [4, 1];

            ArgumentException error = Assert.Throws<ArgumentException>(() =>
                VideoRecipeUtils.EncodeUmT5PromptBatchKeepingPadding(backend, encoder, DModel, Length, Full, tooShort));
            Assert.Equal("tokenSets", error.ParamName);
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    private static T5TextEncoder Encoder(Dictionary<string, Tensor> weights)
    {
        T5TextEncoder encoder = new(new T5TextEncoderConfig
        {
            DModel = DModel, DFf = DFf, DKv = DKv, NumHeads = Heads, NumLayers = Layers, VocabSize = Vocab,
            RelativeAttentionNumBuckets = Buckets,
        });
        encoder.LoadWeights(weights);
        return encoder;
    }

    /// <summary>The tensors the encoder reads for a one-layer gated umT5-shaped model (shared embedding table, one block, final norm).</summary>
    private static Dictionary<string, Tensor> TinyWeights(int seed)
    {
        Tensor R(int r, int c, int s) => LongCatTestData.Random(new TensorShape([r, c]), seed + s, 0.2f);
        Tensor V(int n, int s) => LongCatTestData.Random(new TensorShape([n]), seed + s, 0.2f);
        return new Dictionary<string, Tensor>
        {
            ["shared.weight"] = R(Vocab, DModel, 1),
            ["encoder.final_layer_norm.weight"] = V(DModel, 2),
            ["encoder.block.0.layer.0.layer_norm.weight"] = V(DModel, 3),
            ["encoder.block.0.layer.0.SelfAttention.q.weight"] = R(Heads * DKv, DModel, 4),
            ["encoder.block.0.layer.0.SelfAttention.k.weight"] = R(Heads * DKv, DModel, 5),
            ["encoder.block.0.layer.0.SelfAttention.v.weight"] = R(Heads * DKv, DModel, 6),
            ["encoder.block.0.layer.0.SelfAttention.o.weight"] = R(DModel, Heads * DKv, 7),
            ["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = R(Buckets, Heads, 8),
            ["encoder.block.0.layer.1.layer_norm.weight"] = V(DModel, 9),
            ["encoder.block.0.layer.1.DenseReluDense.wi_0.weight"] = R(DFf, DModel, 10),
            ["encoder.block.0.layer.1.DenseReluDense.wi_1.weight"] = R(DFf, DModel, 11),
            ["encoder.block.0.layer.1.DenseReluDense.wo.weight"] = R(DModel, DFf, 12),
        };
    }
}
