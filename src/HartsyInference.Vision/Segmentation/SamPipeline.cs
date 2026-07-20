using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Segmentation.Sam2;

namespace HartsyInference.Vision.Segmentation;

/// <summary>Segment Anything Model 2 pipeline: image encoder (<see cref="ISamImageEncoder"/>, Hiera by default)
/// + <see cref="SamPromptEncoder"/> + <see cref="SamMaskDecoder"/>. Given a point/box prompt over a
/// model-resolution image it returns candidate masks with IoU (quality) scores. The image is assumed already
/// preprocessed to the model input size (SAM 2 = 1024²) — this pipeline does no resize/normalize.
///
/// <para><b>MobileSAM:</b> MobileSAM reuses this exact prompt encoder + mask decoder and only swaps the
/// <see cref="ISamImageEncoder"/> for a TinyViT backbone. To add it, implement <see cref="ISamImageEncoder"/>
/// with a TinyViT trunk and pass it to the component constructor below — no other pipeline change is needed.</para></summary>
public sealed unsafe class SamPipeline : IVisionPipeline
{
    private readonly IBackend _backend;
    private readonly ISamImageEncoder _encoder;
    private readonly SamPromptEncoder _promptEncoder;
    private readonly SamMaskDecoder _maskDecoder;
    private readonly SafeTensorsLoader? _loader;

    /// <inheritdoc/>
    public string ModelName => "sam2";

    /// <summary>Composes a pipeline from already-constructed components (backbone-agnostic entry point). The
    /// caller owns the component weights; <see cref="Dispose"/> only releases the encoder handle.</summary>
    public SamPipeline(IBackend backend, Sam2Config config, ISamImageEncoder encoder,
        SamPromptEncoder promptEncoder, SamMaskDecoder maskDecoder)
    {
        ArgumentNullException.ThrowIfNull(config);
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _promptEncoder = promptEncoder ?? throw new ArgumentNullException(nameof(promptEncoder));
        _maskDecoder = maskDecoder ?? throw new ArgumentNullException(nameof(maskDecoder));
    }

    private SamPipeline(IBackend backend, Sam2Config config, ISamImageEncoder encoder,
        SamPromptEncoder promptEncoder, SamMaskDecoder maskDecoder, SafeTensorsLoader loader)
        : this(backend, config, encoder, promptEncoder, maskDecoder)
    {
        _loader = loader;
    }

    /// <summary>Loads a SAM 2 (Hiera) checkpoint and builds a ready pipeline. The returned pipeline owns the
    /// checkpoint loader and frees it on <see cref="Dispose"/>.</summary>
    public static SamPipeline Load(IBackend backend, Sam2Config config, string safetensorsPath)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        SafeTensorsLoader loader = null!;
        try
        {
            (Sam2Converter.Groups groups, SafeTensorsLoader ld) = Sam2Converter.LoadAndConvert(safetensorsPath);
            loader = ld;

            HieraImageEncoder encoder = new HieraImageEncoder(config);
            encoder.LoadWeights(groups.ImageEncoder, string.Empty);

            SamPromptEncoder promptEncoder = BuildPromptEncoder(groups.PromptEncoder);

            SamMaskDecoder maskDecoder = new SamMaskDecoder(config);
            maskDecoder.LoadWeights(groups.MaskDecoder, string.Empty);

            return new SamPipeline(backend, config, encoder, promptEncoder, maskDecoder, loader);
        }
        catch (Exception ex)
        {
            Logs.Error($"Failed to load SAM 2 pipeline from '{safetensorsPath}'.", ex);
            loader?.Dispose();
            throw;
        }
    }

    /// <summary>Builds the sparse prompt encoder from the <c>sam_prompt_encoder.*</c> weight group.</summary>
    public static SamPromptEncoder BuildPromptEncoder(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        Tensor gaussian = Require(weights, "pe_layer.positional_encoding_gaussian_matrix");
        SamPositionalEncoding pe = new SamPositionalEncoding(gaussian);

        Tensor[] points = new Tensor[4];
        for (int i = 0; i < 4; i++)
        {
            points[i] = Require(weights, $"point_embeddings.{i}.weight");
        }
        Tensor notAPoint = Require(weights, "not_a_point_embed.weight");
        weights.TryGetValue("no_mask_embed.weight", out Tensor? noMask);
        return new SamPromptEncoder(pe, points, notAPoint, noMask);
    }

    /// <summary>Segments <paramref name="image"/> (<c>[1,3,H,W]</c>, model-resolution) for <paramref name="prompt"/>.
    /// Returns candidate masks at <paramref name="imageW"/>×<paramref name="imageH"/>, best (highest IoU) first.</summary>
    public SamMaskResult[] Segment(SamPrompt prompt, Tensor image, int imageW, int imageH)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(image);
        try
        {
            SamImageFeatures features = _encoder.Forward(_backend, image);
            Tensor embedding = features.VisionFeatures; // [1, dim, Hf, Wf]
            int hf = (int)embedding.Shape[2];
            int wf = (int)embedding.Shape[3];

            Tensor densePe = _promptEncoder.BuildDensePositionalEncoding(hf, wf);
            Tensor sparse = _promptEncoder.EncodeSparse(prompt, imageW, imageH);
            Tensor? dense = _promptEncoder.BuildNoMaskDenseEmbedding(hf, wf);

            (Tensor maskLogits, Tensor iouScores) = _maskDecoder.Forward(
                _backend, embedding, densePe, sparse, dense, multimaskOutput: true,
                features.HighResStride4, features.HighResStride8);

            return BuildResults(maskLogits, iouScores, imageW, imageH);
        }
        catch (Exception ex)
        {
            Logs.Error("SAM 2 segmentation failed.", ex);
            throw;
        }
    }

    /// <summary>Upscales each low-res mask logit to the requested size, thresholds at 0, and pairs it with its
    /// IoU score; results are returned best-first.</summary>
    private SamMaskResult[] BuildResults(Tensor maskLogits, Tensor iouScores, int imageW, int imageH)
    {
        int count = (int)maskLogits.Shape[1];
        int mh = (int)maskLogits.Shape[2];
        int mw = (int)maskLogits.Shape[3];
        long plane = (long)mh * mw;
        float* logitsPtr = (float*)maskLogits.DataPointer;
        float* iouPtr = (float*)iouScores.DataPointer;

        SamMaskResult[] results = new SamMaskResult[count];
        for (int m = 0; m < count; m++)
        {
            Tensor srcPlane = new Tensor(new TensorShape(1, 1, mh, mw), DType.F32);
            float* sp = (float*)srcPlane.DataPointer;
            for (long i = 0; i < plane; i++) sp[i] = logitsPtr[m * plane + i];

            Tensor upsized = new Tensor(new TensorShape(1, 1, imageH, imageW), DType.F32);
            _backend.InterpolateBilinear2D(upsized, srcPlane, alignCorners: false);

            float* up = (float*)upsized.DataPointer;
            byte[] mask = new byte[imageW * imageH];
            for (int i = 0; i < mask.Length; i++) mask[i] = up[i] > 0f ? (byte)1 : (byte)0;

            float score = Math.Clamp(iouPtr[m], 0f, 1f);
            results[m] = new SamMaskResult { Mask = mask, Width = imageW, Height = imageH, IoUScore = score };
        }

        Array.Sort(results, static (a, b) => b.IoUScore.CompareTo(a.IoUScore));
        return results;
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        if (!weights.TryGetValue(key, out Tensor? t))
            throw new KeyNotFoundException($"SamPipeline: missing prompt-encoder weight '{key}'.");
        return t;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _encoder.Dispose();
        _loader?.Dispose();
    }
}
