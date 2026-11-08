using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>SAM 3 segmentation head and dot-product scorer: prompt cross-attention on the encoder memory, pixel
/// decoding, per-query mask logits and per-query class logits.</summary>
/// <remarks>
/// Read from segmentation_head: pixel_decoder.conv_layers.{0,1}, pixel_decoder.norms.{0,1}, cross_attend_prompt.*,
/// cross_attn_norm.*, instance_seg_head.*, mask_predictor.mask_embed.*. Read from dot_prod_scoring: all keys.
/// Unused: pixel_decoder.conv_layers.2 and norms.2 (never run), semantic_seg_head.* (not read by the processor),
/// presence_head (absent). Positions are not read; the 72x72 neck level is replaced by the cross-attended memory.
/// </remarks>
public sealed class Sam3SegmentationHead
{
    private const int CrossAttendHeads = 8;

    private readonly Sam3Config _config;
    private readonly Sam3PixelDecoder _pixelDecoder;
    private readonly Sam3MaskPredictor _maskPredictor;
    private readonly Sam3DotProductScorer _scorer;
    private readonly Tensor _crossNormWeight;
    private readonly Tensor _crossNormBias;
    private readonly Tensor _crossInWeight;
    private readonly Tensor _crossInBias;
    private readonly Tensor _crossOutWeight;
    private readonly Tensor _crossOutBias;
    private readonly Tensor _instanceWeight;
    private readonly Tensor _instanceBias;

    /// <summary>Loads the head and scorer weights; key sets are relative to segmentation_head and dot_prod_scoring.</summary>
    public Sam3SegmentationHead(Sam3Config config, IReadOnlyDictionary<string, Tensor> segmentationWeights,
        IReadOnlyDictionary<string, Tensor> scoringWeights)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(segmentationWeights);
        ArgumentNullException.ThrowIfNull(scoringWeights);
        int c = config.DModel;
        if (c % CrossAttendHeads != 0)
        {
            throw new ArgumentException($"DModel {c} is not divisible by {CrossAttendHeads} prompt cross-attention heads.", nameof(config));
        }

        _config = config;
        _crossNormWeight = Sam3WeightAccess.Require(segmentationWeights, "cross_attn_norm.weight", c);
        _crossNormBias = Sam3WeightAccess.Require(segmentationWeights, "cross_attn_norm.bias", c);
        _crossInWeight = Sam3WeightAccess.Require(segmentationWeights, "cross_attend_prompt.in_proj_weight", 3 * c, c);
        _crossInBias = Sam3WeightAccess.Require(segmentationWeights, "cross_attend_prompt.in_proj_bias", 3 * c);
        _crossOutWeight = Sam3WeightAccess.Require(segmentationWeights, "cross_attend_prompt.out_proj.weight", c, c);
        _crossOutBias = Sam3WeightAccess.Require(segmentationWeights, "cross_attend_prompt.out_proj.bias", c);
        _instanceWeight = Sam3WeightAccess.Require(segmentationWeights, "instance_seg_head.weight", c, c, 1, 1);
        _instanceBias = Sam3WeightAccess.Require(segmentationWeights, "instance_seg_head.bias", c);
        _pixelDecoder = new Sam3PixelDecoder(config, segmentationWeights);
        _maskPredictor = new Sam3MaskPredictor(config, segmentationWeights);
        _scorer = new Sam3DotProductScorer(config, scoringWeights);
    }

    /// <summary>Returns mask logits <c>[1, Q, MaskResolution, MaskResolution]</c> and class logits <c>[1, Q]</c>; the caller owns both.</summary>
    public Sam3InstancePredictions Forward(IBackend backend, Sam3VisionFeatures vision, Sam3DecoderOutput decoder)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(vision);
        ArgumentNullException.ThrowIfNull(decoder);
        ValidateVision(vision);
        ValidateDecoder(decoder);

        int c = _config.DModel;
        int grid = _config.PatchGrid;
        int tokens = grid * grid;

        Tensor memoryTokens = NchwToTokens(backend, decoder.EncoderMemory, c, tokens);
        Tensor fusedTokens = CrossAttendPrompt(backend, memoryTokens, decoder.PromptFeatures, decoder.PromptPaddingMask);
        memoryTokens.Dispose();

        Tensor coarse = new Tensor(new TensorShape(1, c, grid, grid), DType.F32);
        backend.Transpose2D(coarse, fusedTokens, tokens, c);
        fusedTokens.Dispose();

        Tensor[] levels = [vision.Features[0], vision.Features[1], coarse];
        Tensor pixelEmbed = _pixelDecoder.Forward(backend, levels);
        coarse.Dispose();

        int maskSize = _config.MaskResolution;
        Tensor instance = new Tensor(new TensorShape(1, c, maskSize, maskSize), DType.F32);
        backend.Conv2D(instance, pixelEmbed, _instanceWeight, _instanceBias, 1, 1, 0, 0);
        pixelEmbed.Dispose();

        Tensor maskLogits = _maskPredictor.Forward(backend, decoder.QueryHidden, instance);
        instance.Dispose();

        Tensor classLogits = _scorer.Forward(backend, decoder.QueryHidden, decoder.PromptFeatures, decoder.PromptPaddingMask);
        return new Sam3InstancePredictions(maskLogits, classLogits);
    }

    /// <summary>Multihead attention of the memory tokens over the prompt, with a residual and no post-norm.</summary>
    private Tensor CrossAttendPrompt(IBackend backend, Tensor tokens, Tensor prompt, bool[] paddingMask)
    {
        int c = _config.DModel;
        int n = (int)tokens.Shape[1];
        int s = (int)prompt.Shape[1];
        int headDim = c / CrossAttendHeads;

        Tensor normed = new Tensor(tokens.Shape, DType.F32);
        backend.LayerNorm(normed, tokens, _crossNormWeight, _crossNormBias, _config.LayerNormEps);
        Tensor q = new Tensor(new TensorShape(1, n, c), DType.F32);
        backend.LinearWeightRows(q, normed, _crossInWeight, _crossInBias, 0, c);
        normed.Dispose();
        Tensor k = new Tensor(new TensorShape(1, s, c), DType.F32);
        backend.LinearWeightRows(k, prompt, _crossInWeight, _crossInBias, c, c);
        Tensor v = new Tensor(new TensorShape(1, s, c), DType.F32);
        backend.LinearWeightRows(v, prompt, _crossInWeight, _crossInBias, 2 * c, c);

        Tensor qHeads = SplitHeads(backend, q, n, headDim);
        q.Dispose();
        Tensor kHeads = SplitHeads(backend, k, s, headDim);
        k.Dispose();
        Tensor vHeads = SplitHeads(backend, v, s, headDim);
        v.Dispose();

        Tensor mask = BuildKeyPaddingMask(paddingMask);
        Tensor attended = new Tensor(qHeads.Shape, DType.F32);
        backend.ScaledDotProductAttention(attended, qHeads, kHeads, vHeads, mask, 1f / MathF.Sqrt(headDim));
        qHeads.Dispose();
        kHeads.Dispose();
        vHeads.Dispose();
        mask.Dispose();

        Tensor merged = new Tensor(new TensorShape(1, n, CrossAttendHeads, headDim), DType.F32);
        backend.Permute0213(merged, attended, CrossAttendHeads, n, headDim);
        attended.Dispose();

        Tensor projected = new Tensor(tokens.Shape, DType.F32);
        backend.Linear(projected, merged.Reshape(new TensorShape(1, n, c)), _crossOutWeight, _crossOutBias);
        merged.Dispose();
        backend.Add(projected, projected, tokens);
        return projected;
    }

    /// <summary>Reorders <c>[1, T, C]</c> into head-major <c>[1, heads, T, headDim]</c>; the caller owns the result.</summary>
    private static Tensor SplitHeads(IBackend backend, Tensor flat, int length, int headDim)
    {
        Tensor heads = new Tensor(new TensorShape(1, CrossAttendHeads, length, headDim), DType.F32);
        backend.Permute0213(heads, flat.Reshape(new TensorShape(1, length, CrossAttendHeads, headDim)), length, CrossAttendHeads, headDim);
        return heads;
    }

    /// <summary>Additive key mask <c>[1, 1, 1, S]</c>: zero for real tokens, negative infinity for padding.</summary>
    private static Tensor BuildKeyPaddingMask(bool[] paddingMask)
    {
        Tensor mask = new Tensor(new TensorShape(1, 1, 1, paddingMask.Length), DType.F32);
        Span<float> values = mask.AsSpan<float>();
        for (int i = 0; i < paddingMask.Length; i++)
        {
            values[i] = paddingMask[i] ? float.NegativeInfinity : 0f;
        }

        return mask;
    }

    /// <summary>Views the encoder memory <c>[1, C, G, G]</c> as tokens <c>[1, G*G, C]</c>; the caller owns the result.</summary>
    private static Tensor NchwToTokens(IBackend backend, Tensor memory, int channels, int tokens)
    {
        Tensor tokenMajor = new Tensor(new TensorShape(1, tokens, channels), DType.F32);
        backend.Transpose2D(tokenMajor, memory.Reshape(new TensorShape(1, channels, tokens)), channels, tokens);
        return tokenMajor;
    }

    private void ValidateVision(Sam3VisionFeatures vision)
    {
        if (vision.Features.Length < 2)
        {
            throw new ArgumentException($"The segmentation head reads FPN levels 0 and 1; got {vision.Features.Length} levels.", nameof(vision));
        }

        int c = _config.DModel;
        int fine = _config.MaskResolution;
        RequireF32(vision.Features[0], "vision.Features[0]", 1, c, fine, fine);
        RequireF32(vision.Features[1], "vision.Features[1]", 1, c, fine / 2, fine / 2);
    }

    private void ValidateDecoder(Sam3DecoderOutput decoder)
    {
        int c = _config.DModel;
        int grid = _config.PatchGrid;
        RequireF32(decoder.EncoderMemory, "decoder.EncoderMemory", 1, c, grid, grid);
        RequireF32(decoder.QueryHidden, "decoder.QueryHidden", 1, _config.NumQueries, c);

        Tensor prompt = decoder.PromptFeatures;
        if (prompt.DType != DType.F32 || prompt.Shape.Rank != 3 || prompt.Shape[0] != 1 || prompt.Shape[2] != c)
        {
            throw new ArgumentException($"decoder.PromptFeatures must be F32 [1, S, {c}]; got {prompt.DType} {prompt.Shape}.", nameof(decoder));
        }

        ArgumentNullException.ThrowIfNull(decoder.PromptPaddingMask);
        if (decoder.PromptPaddingMask.Length != prompt.Shape[1])
        {
            throw new ArgumentException($"decoder.PromptPaddingMask has {decoder.PromptPaddingMask.Length} entries for {prompt.Shape[1]} tokens.", nameof(decoder));
        }

        if (Array.IndexOf(decoder.PromptPaddingMask, false) < 0)
        {
            throw new ArgumentException("decoder.PromptPaddingMask marks every prompt token as padding.", nameof(decoder));
        }
    }

    private static void RequireF32(Tensor tensor, string name, params long[] expectedShape)
    {
        TensorShape expected = new(expectedShape);
        if (tensor.DType != DType.F32 || tensor.Shape != expected)
        {
            throw new ArgumentException($"{name} must be F32 {expected}; got {tensor.DType} {tensor.Shape}.");
        }
    }
}
