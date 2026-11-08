using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Segmentation.Sam2;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>SAM 3 mask predictor: a 3-layer MLP maps query states to mask embeddings, which dot with the pixel embedding.</summary>
public sealed class Sam3MaskPredictor
{
    private const string Prefix = "mask_predictor.mask_embed.";

    private readonly SamMlpHead _maskEmbed;
    private readonly int _channels;

    /// <summary>Loads the mask-embedding MLP; the layer count comes from the config and extra layers are rejected.</summary>
    public Sam3MaskPredictor(Sam3Config config, IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        _channels = config.DModel;
        for (int i = 0; i < config.MaskEmbedLayers; i++)
        {
            Sam3WeightAccess.Require(weights, $"{Prefix}layers.{i}.weight", _channels, _channels);
            Sam3WeightAccess.Require(weights, $"{Prefix}layers.{i}.bias", _channels);
        }

        if (weights.ContainsKey($"{Prefix}layers.{config.MaskEmbedLayers}.weight"))
        {
            throw new InvalidDataException($"SAM 3 mask predictor has more than {config.MaskEmbedLayers} layers.");
        }

        _maskEmbed = new SamMlpHead();
        _maskEmbed.LoadWeights(weights, Prefix);
    }

    /// <summary>Returns mask logits <c>[1, Q, H, W]</c> from queries <c>[1, Q, C]</c> and a pixel embedding <c>[1, C, H, W]</c>; the caller owns the result.</summary>
    public Tensor Forward(IBackend backend, Tensor queries, Tensor pixelEmbed)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (queries.DType != DType.F32 || queries.Shape.Rank != 3 || queries.Shape[0] != 1 || queries.Shape[2] != _channels)
        {
            throw new ArgumentException($"Mask predictor queries must be F32 [1, Q, {_channels}]; got {queries.DType} {queries.Shape}.", nameof(queries));
        }

        if (pixelEmbed.DType != DType.F32 || pixelEmbed.Shape.Rank != 4 || pixelEmbed.Shape[0] != 1 || pixelEmbed.Shape[1] != _channels)
        {
            throw new ArgumentException($"Mask predictor pixel embedding must be F32 [1, {_channels}, H, W]; got {pixelEmbed.DType} {pixelEmbed.Shape}.", nameof(pixelEmbed));
        }

        int numQueries = (int)queries.Shape[1];
        int height = (int)pixelEmbed.Shape[2];
        int width = (int)pixelEmbed.Shape[3];
        long pixels = (long)height * width;

        Tensor flatQueries = queries.Reshape(new TensorShape(numQueries, _channels));
        Tensor embeds = _maskEmbed.Forward(backend, flatQueries);

        // The GEMM writes an owned flat result; the 4D view is taken afterwards, and Reshape reads the result first.
        Tensor outputFlat = new Tensor(new TensorShape(1, numQueries, pixels), DType.F32);
        Tensor embedsBatched = embeds.Reshape(new TensorShape(1, numQueries, _channels));
        Tensor pixelsBatched = pixelEmbed.Reshape(new TensorShape(1, _channels, pixels));
        backend.BatchedMatMul(outputFlat, embedsBatched, pixelsBatched);
        embeds.Dispose();
        return outputFlat.Reshape(new TensorShape(1, numQueries, height, width));
    }
}
