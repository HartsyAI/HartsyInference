using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>SAM 3 detector transformer for text prompts (batch 1, F32, batch-first; equal to upstream sequence-first).</summary>
public sealed class Sam3Transformer
{
    private readonly Sam3Config _config;
    private readonly Sam3GeometryEncoder _geometry;
    private readonly Sam3FusionEncoder _encoder;
    private readonly Sam3DetectorDecoder _decoder;

    /// <summary>Builds the transformer and loads its weights; fails naming the first missing key. Key sets are relative to
    /// <c>detector.transformer.encoder.</c>, <c>detector.transformer.decoder.</c> and <c>detector.geometry_encoder.</c>.</summary>
    public Sam3Transformer(
        Sam3Config config,
        IReadOnlyDictionary<string, Tensor> fusionWeights,
        IReadOnlyDictionary<string, Tensor> decoderWeights,
        IReadOnlyDictionary<string, Tensor> geometryWeights)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(fusionWeights);
        ArgumentNullException.ThrowIfNull(decoderWeights);
        ArgumentNullException.ThrowIfNull(geometryWeights);
        ValidateConfig(config);

        _config = config;
        _geometry = new Sam3GeometryEncoder(config);
        _encoder = new Sam3FusionEncoder(config);
        _decoder = new Sam3DetectorDecoder(config);
        _encoder.LoadWeights(fusionWeights);
        _decoder.LoadWeights(decoderWeights);
        _geometry.LoadWeights(geometryWeights);
    }

    /// <summary>Runs the detector transformer on the last (72x72) vision level and the text memory. Returns owned tensors;
    /// <see cref="Sam3DecoderOutput.PromptFeatures"/> is a new tensor, not the text input.</summary>
    public Sam3DecoderOutput Forward(IBackend backend, Sam3VisionFeatures vision, Sam3TextFeatures text)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(vision);
        ArgumentNullException.ThrowIfNull(text);

        int grid = _config.PatchGrid;
        int positions = grid * grid;
        int d = _config.DModel;
        int textLength = _config.TextContextLength;
        if (vision.Features.Length == 0 || vision.Positions.Length != vision.Features.Length)
            throw new ArgumentException("Vision features and positions must be non-empty and the same length.", nameof(vision));
        Tensor feature = vision.Features[^1];
        Tensor position = vision.Positions[^1];
        Sam3Ops.RequireShape(feature, "vision feature", 1, d, grid, grid);
        Sam3Ops.RequireShape(position, "vision position", 1, d, grid, grid);
        Sam3Ops.RequireShape(text.Memory, "text memory", 1, textLength, d);
        if (text.PaddingMask.Length != textLength)
            throw new ArgumentException($"Text padding mask has {text.PaddingMask.Length} entries, expected {textLength}.", nameof(text));

        Tensor imageTokens = FlattenGrid(backend, feature, d, positions);
        Tensor imagePos = FlattenGrid(backend, position, d, positions);

        Tensor geometry = _geometry.Forward(backend, imageTokens, imagePos);
        Tensor prompt = Sam3Ops.ConcatRows(backend, text.Memory, geometry);
        geometry.Dispose();
        bool[] promptPadding = new bool[textLength + 1];
        Array.Copy(text.PaddingMask, promptPadding, textLength);

        Tensor memory = _encoder.Forward(backend, imageTokens, imagePos, prompt, promptPadding);
        imageTokens.Dispose();

        (Tensor Boxes, Tensor QueryHidden, Tensor PresenceLogit) decoded = _decoder.Forward(backend, memory, imagePos, prompt, promptPadding);
        imagePos.Dispose();

        Tensor encoderMemory = Sam3Ops.NewF32(1, d, grid, grid);
        backend.Transpose2D(encoderMemory, memory, positions, d);
        memory.Dispose();

        return new Sam3DecoderOutput(decoded.Boxes, decoded.QueryHidden, decoded.PresenceLogit, encoderMemory, prompt, promptPadding);
    }

    /// <summary>Converts a channel-major <c>[1, C, H, W]</c> map into a token-major <c>[1, H*W, C]</c> sequence (row index h*W + w).</summary>
    private static Tensor FlattenGrid(IBackend backend, Tensor feature, int channels, int positions)
    {
        Tensor tokens = Sam3Ops.NewF32(1, positions, channels);
        backend.Transpose2D(tokens, feature, channels, positions);
        return tokens;
    }

    private static void ValidateConfig(Sam3Config config)
    {
        if (config.DModel <= 0 || config.DModel % 2 != 0)
            throw new ArgumentException("DModel must be a positive even width.", nameof(config));
        if (config.EncoderHeads <= 0 || config.DModel % config.EncoderHeads != 0)
            throw new ArgumentException("DModel must divide evenly into EncoderHeads.", nameof(config));
        if (config.DecoderHeads <= 0 || config.DModel % config.DecoderHeads != 0)
            throw new ArgumentException("DModel must divide evenly into DecoderHeads.", nameof(config));
        if (config.EncoderLayers < 1 || config.DecoderLayers < 1 || config.GeometryLayers < 1)
            throw new ArgumentException("Encoder, decoder and geometry layer counts must be at least one.", nameof(config));
        if (config.EncoderFfn <= 0 || config.DecoderFfn <= 0)
            throw new ArgumentException("Feed-forward widths must be positive.", nameof(config));
        if (config.NumQueries < 1 || config.TextContextLength < 1)
            throw new ArgumentException("NumQueries and TextContextLength must be at least one.", nameof(config));
        if (config.PatchSize <= 0 || config.ImageSize <= 0 || config.ImageSize % config.PatchSize != 0)
            throw new ArgumentException("ImageSize must be a positive multiple of PatchSize.", nameof(config));
        if (config.LayerNormEps <= 0f)
            throw new ArgumentException("LayerNormEps must be positive.", nameof(config));
    }
}
