using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The side models of the LongCat recipe tests: a tiny umT5 and a <see cref="LongCatStandInVae"/>, in place of the downloads.</summary>
internal sealed class LongCatRecipeStandIns : IDisposable
{
    private const int UmT5Vocab = 256384, UmT5Buckets = 32, Tokens = 8;

    private readonly List<Dictionary<string, Tensor>> _umt5Weights = [];

    /// <summary>The stand-ins for a DiT of <paramref name="captionChannels"/> caption width and <paramref name="latentChannels"/> latent channels.</summary>
    public LongCatRecipeStandIns(int captionChannels, int latentChannels)
    {
        Vae = new LongCatStandInVae(latentChannels);
        Sides = new LongCatSideModels(
            () =>
            {
                (T5TextEncoder encoder, Dictionary<string, Tensor> weights) = TinyUmT5(captionChannels, seed: 72);
                _umt5Weights.Add(weights);
                return (encoder, T5Tokenizer.CreateUmt5(maxLength: Tokens));
            },
            () => (Vae, Vae));
    }

    public LongCatSideModels Sides { get; }

    public LongCatStandInVae Vae { get; }

    /// <summary>Disposes the tiny umT5's weights. The encoder reads them while it lives, so dispose this after the pipeline.</summary>
    public void Dispose()
    {
        foreach (Dictionary<string, Tensor> weights in _umt5Weights)
            LongCatTestData.DisposeAll(weights);
        _umt5Weights.Clear();
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
}

/// <summary>Tensors of the Wan VAE's shapes without weights. Each latent frame decodes to a flat colour that follows its mean, a clip encodes to a
/// latent filled with its mean, and the calls are recorded.</summary>
internal sealed class LongCatStandInVae(int channels) : IWanVaeEncoder, IWanVaeDecoder
{
    /// <summary>The latents handed to <see cref="Decode"/>, in call order, with their latent frame counts.</summary>
    public List<(int Frames, float[] Values)> Decoded { get; } = [];

    /// <summary>The frame counts of the clips handed to <see cref="Encode"/>, in call order.</summary>
    public List<int> EncodedClipFrames { get; } = [];

    public Tensor Encode(IBackend backend, Tensor rgbClip)
    {
        EncodedClipFrames.Add((int)rgbClip.Shape[2]);
        Tensor latent = new(new TensorShape([1, channels, (rgbClip.Shape[2] - 1) / 4 + 1, rgbClip.Shape[3] / 8, rgbClip.Shape[4] / 8]), DType.F32);
        latent.AsSpan<float>().Fill(Mean(rgbClip.AsReadOnlySpan<float>()));
        return latent;
    }

    public Tensor Decode(IBackend backend, Tensor latent)
    {
        int frames = (int)latent.Shape[2], pixels = 1 + 4 * (frames - 1);
        float[] values = latent.AsReadOnlySpan<float>().ToArray();
        Decoded.Add((frames, values));
        int height = (int)latent.Shape[3] * 8, width = (int)latent.Shape[4] * 8, perFrame = values.Length / (channels * frames);
        Tensor rgb = new(new TensorShape([1, 3, pixels, height, width]), DType.F32);
        Span<float> output = rgb.AsSpan<float>();
        for (int p = 0; p < pixels; p++)
        {
            int from = (p + 3) / 4;
            float sum = 0;
            for (int c = 0; c < channels; c++)
                sum += Mean(values.AsSpan((c * frames + from) * perFrame, perFrame));
            for (int colour = 0; colour < 3; colour++)
                output.Slice((colour * pixels + p) * height * width, height * width).Fill(MathF.Tanh(sum / channels));
        }
        return rgb;
    }

    public Tensor EncodeRgbFrame(IBackend backend, ReadOnlySpan<byte> rgb24, int width, int height) =>
        throw new NotSupportedException("The LongCat recipes encode clips only.");

    public IEnumerable<Tensor> EnumerateWeights() => [];

    private static float Mean(ReadOnlySpan<float> values)
    {
        double sum = 0;
        foreach (float value in values)
            sum += value;
        return (float)(sum / values.Length);
    }
}
