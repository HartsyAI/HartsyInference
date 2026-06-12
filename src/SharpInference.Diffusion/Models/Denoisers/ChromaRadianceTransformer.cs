using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Chroma Radiance transformer (<c>lodestones/Chroma1-Radiance</c>) — the unmodified Chroma backbone
/// operating directly in pixel space. Composition: <see cref="ChromaRadianceImagePatchifier"/> (Conv2d 3→3072, k16/s16,
/// replaces VAE + <c>img_in</c>) → <see cref="ChromaTransformer.ForwardCore"/> (19 double + 38 single blocks, byte-identical
/// weights to classic Chroma) → <see cref="ChromaRadianceNerfHead"/> (per-patch hypernetwork, replaces <c>final_layer</c>).
/// Predicts <b>x0</b> [B, 3, H, W] in [-1, 1]; the pipeline converts to velocity via
/// <c>SharpInference.Diffusion.Utilities.X0Prediction.ToVelocity</c>. No final modulation norm is applied before the
/// NeRF head (verified against ComfyUI <c>ChromaRadiance.forward</c>); the modulation table's last two rows go unused.</summary>
public sealed class ChromaRadianceTransformer : IDisposable
{
    private readonly ChromaTransformer _backbone;
    private readonly ChromaRadianceImagePatchifier _patchifier;
    private readonly ChromaRadianceNerfHead _nerfHead;
    private int _disposed;

    /// <summary>Creates a Radiance transformer from configuration (use <see cref="ChromaRadianceConfig.FromWeights"/>).</summary>
    public ChromaRadianceTransformer(ChromaRadianceConfig config)
    {
        _backbone = new ChromaTransformer(config.Backbone);
        _patchifier = new ChromaRadianceImagePatchifier();
        _nerfHead = new ChromaRadianceNerfHead(
            config.PatchSize, config.NerfHidden, config.MaxFreqs, config.NerfDepth, config.NerfMlpRatio);
    }

    /// <summary>Pixel patch size (16 for the published release; read from the conv weight on load).</summary>
    public int PatchSize => _patchifier.PatchSize;

    /// <summary>Loads all weights from a converted Radiance dict: backbone keys are classic-Chroma diffusers naming;
    /// <c>img_in_patch.*</c> / <c>nerf_*</c> keys pass through the converter verbatim. Radiance checkpoints have no
    /// <c>x_embedder.*</c> / <c>proj_out.*</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _backbone.LoadWeightsInternal(weights, requireImageProjections: false);
        _patchifier.LoadWeights(weights);
        _nerfHead.LoadWeights(weights);
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in _backbone.EnumerateWeights()) yield return w;
        foreach (Tensor w in _patchifier.EnumerateWeights()) yield return w;
        foreach (Tensor w in _nerfHead.EnumerateWeights()) yield return w;
    }

    /// <summary>Forward pass: predicts the clean image (x0) for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="rgb">Noisy pixel sample [B, 3, H, W] in [-1, 1]; H/W must be multiples of <see cref="PatchSize"/>.</param>
    /// <param name="encoderHidden">T5 text embeddings [B, txtSeqLen, 4096].</param>
    /// <param name="timestep">Flow-match time in [0, 1] (pipeline passes sigma directly, as classic Chroma).</param>
    /// <param name="txtSeqLen">Number of text tokens.</param>
    /// <param name="attentionMask">Optional [B, txtSeqLen] mask (Chroma "first padding token unmasked" rule).</param>
    /// <returns>x0 prediction [B, 3, H, W].</returns>
    public Tensor Forward(
        IBackend backend,
        Tensor rgb,
        Tensor encoderHidden,
        float timestep,
        int txtSeqLen,
        Tensor? attentionMask)
    {
        int height = (int)rgb.Shape[2];
        int width = (int)rgb.Shape[3];
        int patch = _patchifier.PatchSize;
        int hPacked = height / patch;
        int wPacked = width / patch;

        Tensor tokens = _patchifier.Forward(backend, rgb);

        // ForwardCore consumes (disposes) the token tensor and returns pre-final-norm img tokens + the modulation
        // table. Radiance never applies the final modulation rows — the NeRF head replaces final_layer entirely.
        (Tensor imgOut, Tensor modTable) = _backbone.ForwardCore(
            backend, tokens, encoderHidden, timestep, txtSeqLen, hPacked, wPacked, attentionMask);
        modTable.Dispose();

        Tensor x0 = _nerfHead.Forward(backend, rgb, imgOut);
        imgOut.Dispose();
        return x0;
    }

    /// <summary>Releases component weight references.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _backbone.Dispose();
            _patchifier.Dispose();
            _nerfHead.Dispose();
        }
        GC.SuppressFinalize(this);
    }
}
