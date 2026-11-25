using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The base LongCat recipe run end to end on a tiny checkpoint written to disk: the recipe loads the DiT from the folder, takes umT5 and
/// the VAE from injected stand-ins (no side-model downloads), encodes the prompt, denoises, and decodes frames. The values are not meaningful; the
/// test checks that the path runs and returns the requested clip. The real umT5 and VAE are checked at the real-weight run (B6/B8).</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatRecipeEndToEndTests
{
    private const int UmT5Vocab = 256384, UmT5Buckets = 32, Tokens = 8;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheBaseRecipeGeneratesTheRequestedClipFromATinyCheckpoint(bool int8)
    {
        List<Dictionary<string, Tensor>> umt5Weights = [];
        string directory = Path.Combine(Path.GetTempPath(), "longcat-e2e-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            Dictionary<string, Tensor> weights = LongCatTestData.Weights(LongCatTestData.Tiny, seed: 71);
            if (int8)
            {
                // The release's per-row INT8 layout: the quantized linears carry .weight_int8 and .weight_scale, under the INT8 index name.
                Dictionary<string, Tensor> quantized = LongCatTestData.QuantizePerRow(weights);
                LongCatTestData.WriteCheckpoint(directory, quantized, indexFileName: LongCatCheckpoint.Int8IndexFileName);
                foreach ((string key, Tensor tensor) in quantized)
                {
                    if (!weights.TryGetValue(key, out Tensor? dense) || !ReferenceEquals(dense, tensor))
                        tensor.Dispose();
                }
            }
            else
            {
                LongCatTestData.WriteCheckpoint(directory, weights);
            }
            LongCatTestData.DisposeAll(weights);

            // The encoder reads its weight tensors for as long as it lives, so they are disposed only after the pipeline is.
            LongCatSideModels sides = new(
                () =>
                {
                    (T5TextEncoder encoder, Dictionary<string, Tensor> weights) = TinyUmT5(LongCatTestData.Tiny.CaptionChannels, seed: 72);
                    umt5Weights.Add(weights);
                    return (encoder, T5Tokenizer.CreateUmt5(maxLength: Tokens));
                },
                () => { StandIn vae = new(LongCatTestData.Tiny.InChannels); return (vae, vae); });

            using CpuBackend backend = new();
            IVideoRecipePipeline pipeline = new LongCatVideoRecipe(sides).Construct(new RecipeContext
            {
                CheckpointPath = directory, Backend = backend,
            });
            try
            {
                VideoRequest request = new()
                {
                    Prompt = "a lighthouse at dusk", NegativePrompt = "", Seed = 3,  
                };
                VideoEffectiveSettings settings = new()
                {
                    Width = 16, Height = 16, Frames = 5, Fps = 8, Steps = 2, CfgScale = 1f, Seed = 3,
                    ReferenceSizing = VideoReferenceSizing.Native, LockedFields = default,
                };

                VideoGenerationResult result = pipeline.Generate(request, settings, progress: null, CancellationToken.None);

                Assert.Equal(5, result.Frames.Count);
                Assert.All(result.Frames, frame => Assert.NotNull(frame));
            }
            finally
            {
                pipeline.Dispose();
            }
        }
        finally
        {
            foreach (Dictionary<string, Tensor> weights in umt5Weights)
                LongCatTestData.DisposeAll(weights);
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>A tiny umT5 encoder whose width is the DiT's caption width, with the real 256k vocabulary so the tokenizer's ids are in range.
    /// The weights are returned, not disposed: the encoder reads them while it lives.</summary>
    private static (T5TextEncoder Encoder, Dictionary<string, Tensor> Weights) TinyUmT5(int dModel, int seed)
    {
        const int heads = 2, dKv = 8, dFf = 2 * 48;
        Tensor R(int r, int c, int s) => LongCatTestData.Random(new TensorShape([r, c]), seed + s, 0.2f);
        Tensor V(int n, int s) => LongCatTestData.Random(new TensorShape([n]), seed + s, 0.2f);
        Dictionary<string, Tensor> weights = new()
        {
            ["shared.weight"] = R(UmT5Vocab, dModel, 1),
            ["encoder.final_layer_norm.weight"] = V(dModel, 2),
            ["encoder.block.0.layer.0.layer_norm.weight"] = V(dModel, 3),
            ["encoder.block.0.layer.0.SelfAttention.q.weight"] = R(heads * dKv, dModel, 4),
            ["encoder.block.0.layer.0.SelfAttention.k.weight"] = R(heads * dKv, dModel, 5),
            ["encoder.block.0.layer.0.SelfAttention.v.weight"] = R(heads * dKv, dModel, 6),
            ["encoder.block.0.layer.0.SelfAttention.o.weight"] = R(dModel, heads * dKv, 7),
            ["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = R(UmT5Buckets, heads, 8),
            ["encoder.block.0.layer.1.layer_norm.weight"] = V(dModel, 9),
            ["encoder.block.0.layer.1.DenseReluDense.wi_0.weight"] = R(dFf, dModel, 10),
            ["encoder.block.0.layer.1.DenseReluDense.wi_1.weight"] = R(dFf, dModel, 11),
            ["encoder.block.0.layer.1.DenseReluDense.wo.weight"] = R(dModel, dFf, 12),
        };
        T5TextEncoder encoder = new(new T5TextEncoderConfig
        {
            DModel = dModel, DFf = dFf, DKv = dKv, NumHeads = heads, NumLayers = 1, VocabSize = UmT5Vocab,
            RelativeAttentionNumBuckets = UmT5Buckets,
        });
        encoder.LoadWeights(weights);
        return (encoder, weights);
    }

    /// <summary>Returns tensors of the Wan VAE's shapes for a latent with <paramref name="channels"/> channels, without weights.</summary>
    private sealed class StandIn : IWanVaeEncoder, IWanVaeDecoder
    {
        private readonly int _channels;

        public StandIn(int channels) => _channels = channels;

        public Tensor Encode(IBackend backend, Tensor rgbClip) =>
            new(new TensorShape([1, _channels, (rgbClip.Shape[2] - 1) / 4 + 1, rgbClip.Shape[3] / 8, rgbClip.Shape[4] / 8]), DType.F32);

        public Tensor Decode(IBackend backend, Tensor latent) =>
            new(new TensorShape([1, 3, 1 + 4 * (latent.Shape[2] - 1), latent.Shape[3] * 8, latent.Shape[4] * 8]), DType.F32);

        public Tensor EncodeRgbFrame(IBackend backend, ReadOnlySpan<byte> rgb24, int width, int height) =>
            throw new NotSupportedException("The end-to-end test uses the clip path only.");

        public IEnumerable<Tensor> EnumerateWeights() => [];
    }
}
