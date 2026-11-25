using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The umT5 batch encode that keeps padded positions (the LongCat text seam) on a tiny random encoder: every row of every context comes
/// out of the encoder, and the padded rows are kept non-zero, where the Wan path would zero them. No pretrained weights are used.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class UmT5KeepPaddingTests
{
    private const int DModel = 16, Heads = 2, DKv = 8, DFf = 32, Layers = 1, Vocab = 20, Buckets = 32, Length = 4;

    [Fact]
    public void PaddedRowsAreKeptNonZeroAndEachContextHasItsOwnRows()
    {
        Dictionary<string, Tensor> weights = TinyWeights(seed: 61);
        try
        {
            T5TextEncoderConfig config = new()
            {
                DModel = DModel, DFf = DFf, DKv = DKv, NumHeads = Heads, NumLayers = Layers, VocabSize = Vocab,
                RelativeAttentionNumBuckets = Buckets,
            };
            using T5TextEncoder encoder = new(config);
            encoder.LoadWeights(weights);
            using CpuBackend backend = new();

            // Token 0 is the padding id. The second context has two real tokens and two pad positions.
            int[] full = [5, 6, 7, 1];
            int[] padded = [4, 1, 0, 0];
            Tensor[] embeds = VideoRecipeUtils.EncodeUmT5PromptBatchKeepingPadding(backend, encoder, DModel, Length, full, padded);

            Assert.Equal(2, embeds.Length);
            Assert.All(embeds, e => Assert.Equal(Length * DModel, (int)e.ElementCount));
            // The padded positions are the last two rows of the second context: they are kept, not zeroed.
            float[] paddedRows = embeds[1].AsReadOnlySpan<float>()[(2 * DModel)..].ToArray();
            Assert.Contains(paddedRows, v => v != 0f);
            // The two contexts are encoded in one batch and must not share rows.
            Assert.NotEqual(embeds[0].AsReadOnlySpan<float>()[..DModel].ToArray(), embeds[1].AsReadOnlySpan<float>()[..DModel].ToArray());
            Assert.All(embeds[0].AsReadOnlySpan<float>().ToArray(), v => Assert.True(float.IsFinite(v)));
            Assert.All(embeds[1].AsReadOnlySpan<float>().ToArray(), v => Assert.True(float.IsFinite(v)));

            foreach (Tensor e in embeds)
                e.Dispose();
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
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
