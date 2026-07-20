using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Vision.Siglip;
using ImagePromptRequest = HartsyInference.Engine.Requests.IpAdapter;

namespace HartsyInference.Engine.Features;

/// <summary>Resolves a FLUX.1 Redux style model into conditioning tokens: SigLIP so400m/14-384 patch tokens → Redux
/// projector → <c>[1, 729, 4096]</c> tokens the Flux pipeline appends after the T5 stream. Redux shares the request's
/// image-prompt slot, so the reference image comes from <see cref="ImagePromptRequest.PromptImages"/>.
///
/// <para>Strength semantics match ComfyUI/SwarmUI's workflow generation: the multiply strength scales the redux tokens
/// (<c>StyleModelApply strength_type=multiply</c>), and the merge strength — a ConditioningAverage between styled and
/// original conditioning — reduces to ANOTHER scalar on the redux tokens, because the text tokens are identical on both
/// sides and the original's missing redux positions are zero-padded. The apply-start fraction maps to the pipeline's
/// per-step conditioning switch.</para></summary>
public static class ReduxResolver
{
    /// <summary>Resolved Redux conditioning: projected style tokens (strength pre-applied) plus the apply-start fraction.</summary>
    public sealed class ReduxSpec : IDisposable
    {
        /// <summary>The projected style tokens.</summary>
        public required Tensor Embeds { get; init; }

        /// <summary>Schedule fraction at which the style tokens start applying.</summary>
        public required float ApplyStart { get; init; }

        /// <summary>Frees the token tensor.</summary>
        public void Dispose() => Embeds?.Dispose();
    }

    /// <summary>Models-root-relative folders searched for the Redux projector.</summary>
    private static readonly string[] _styleFolders = ["style_models", "StyleModels", "style_model"];

    private static readonly object _cacheLock = new object();
    private static string? _cachedKey;
    private static SiglipVisionEncoder? _cachedSiglip;
    private static ReduxImageEncoder? _cachedProjector;
    // The encoders' weight tensors are views over these loaders' mmaps — the loaders must outlive the cached encoders.
    private static SafeTensorsLoader? _cachedVisionLoader;
    private static SafeTensorsLoader? _cachedStyleLoader;

    /// <summary>Resolves the Redux tokens, or null when no style model is selected. Throws with a clear message when a
    /// style model is named but unusable (missing file / missing prompt image). Caller disposes the returned spec.</summary>
    /// <param name="styleModel">Redux projector model id or path; null/"None" disables Redux.</param>
    /// <param name="visionModel">CLIP/SigLIP vision tower override; null auto-downloads <see cref="SideModels.SigclipVision384"/>.</param>
    public static async Task<ReduxSpec?> ResolveAsync(
        ImagePromptRequest? imagePrompt,
        string? styleModel,
        string? visionModel,
        double multiplyStrength,
        double mergeStrength,
        double applyStartFraction,
        IBackend backend,
        Action<string> log,
        CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(log);
        if (string.IsNullOrWhiteSpace(styleModel) || styleModel == "None")
        {
            return null;
        }
        if (imagePrompt?.PromptImages is null || imagePrompt.PromptImages.Count == 0)
        {
            throw new InvalidOperationException(
                "A style model (FLUX.1 Redux) is selected but no prompt image was provided. "
                + "Add the style reference via the image-prompt input or clear the style model.");
        }
        float multiply = (float)multiplyStrength;
        float merge = (float)Math.Clamp(mergeStrength, 0.0, 1.0);
        float applyStart = (float)Math.Clamp(applyStartFraction, 0.0, 1.0);
        float scale = multiply * merge;

        string styleModelPath = ModelFileLocator.Require(styleModel, "Style model", _styleFolders);
        string visionPath = ModelFileLocator.Find(visionModel, "clip_vision", "text_encoders")
            ?? await ModelDownloader.EnsureSideModelAsync(SideModels.SigclipVision384, null, ct).ConfigureAwait(false);

        (SiglipVisionEncoder siglip, ReduxImageEncoder projector) = GetOrLoad(styleModelPath, visionPath, log);

        // SigLIP encode of the FIRST prompt image (ComfyUI's StyleModelApply is also single-image). Straight stretch to
        // 384×384 (crop:"none", matching the redux CLIPVisionEncode wiring) + mean/std 0.5 normalize.
        Tensor pixels = PreprocessSiglip(imagePrompt.PromptImages[0], siglip.Preset.ImageSize);
        Tensor hidden;
        try
        {
            hidden = siglip.EncodeHiddenStates(backend, pixels);
        }
        finally
        {
            pixels.Dispose();
        }
        Tensor embeds;
        try
        {
            embeds = projector.Project(backend, hidden);
        }
        finally
        {
            hidden.Dispose();
        }

        // Free the encoder weights from the device — the DiT needs the VRAM; the next resolve re-uploads on demand.
        backend.FreeWeights(siglip.EnumerateWeights());

        if (scale != 1.0f)
        {
            ScaleInPlace(embeds, scale);
        }
        // Host-materialize so the tokens survive the pipeline's pre-loop activation sweep.
        Materialize(embeds);
        log($"Redux style tokens ready: {embeds.Shape[1]} tokens, scale={scale:F2} (multiply {multiply:F2} × merge {merge:F2}), applyStart={applyStart:F2}.");
        return new ReduxSpec { Embeds = embeds, ApplyStart = applyStart };
    }

    private static (SiglipVisionEncoder, ReduxImageEncoder) GetOrLoad(string styleModelPath, string visionPath, Action<string> log)
    {
        string key = styleModelPath + "|" + visionPath;
        lock (_cacheLock)
        {
            if (_cachedKey == key && _cachedSiglip is not null && _cachedProjector is not null)
            {
                log("Redux encoders (cached).");
                return (_cachedSiglip, _cachedProjector);
            }
            _cachedProjector?.Dispose();
            _cachedVisionLoader?.Dispose();
            _cachedStyleLoader?.Dispose();
            _cachedKey = null;
            _cachedSiglip = null;
            _cachedProjector = null;
            _cachedVisionLoader = null;
            _cachedStyleLoader = null;

            log($"Loading SigLIP vision encoder: {Path.GetFileName(visionPath)}");
            SafeTensorsLoader visionLoader = new SafeTensorsLoader();
            visionLoader.Load(visionPath);
            SiglipVisionEncoder siglip = new SiglipVisionEncoder(SiglipPreset.So400m14_384);
            siglip.LoadWeights(visionLoader.GetAllTensors());

            log($"Loading Redux projector: {Path.GetFileName(styleModelPath)}");
            SafeTensorsLoader styleLoader = new SafeTensorsLoader();
            styleLoader.Load(styleModelPath);
            ReduxImageEncoder projector = new ReduxImageEncoder();
            projector.LoadWeights(styleLoader.GetAllTensors());

            _cachedKey = key;
            _cachedSiglip = siglip;
            _cachedProjector = projector;
            _cachedVisionLoader = visionLoader;
            _cachedStyleLoader = styleLoader;
            return (siglip, projector);
        }
    }

    /// <summary>Stretch-resize to <paramref name="imageSize"/>² (no crop — diffusers' SiglipImageProcessor and the
    /// crop:"none" CLIPVisionEncode wiring both stretch) plus SigLIP normalize (mean = std = 0.5).</summary>
    private static Tensor PreprocessSiglip(Requests.ImageData input, int imageSize)
    {
        byte[] rgb = FeatureImaging.ResizeRgb24(input, imageSize, imageSize);
        return FeatureImaging.RgbToTensorMinusOneOne(rgb, imageSize, imageSize);
    }

    /// <summary>Forces a host copy of a possibly device-resident tensor so it survives the pipeline's activation sweep.</summary>
    private static unsafe void Materialize(Tensor t) => _ = t.DataPointer;

    private static unsafe void ScaleInPlace(Tensor t, float scale)
    {
        float* p = (float*)t.DataPointer;
        long count = t.ElementCount;
        for (long i = 0; i < count; i++)
        {
            p[i] *= scale;
        }
    }
}
