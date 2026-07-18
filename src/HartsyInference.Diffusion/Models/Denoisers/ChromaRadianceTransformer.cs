using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Chroma Radiance transformer (<c>lodestones/Chroma1-Radiance</c>) — the unmodified Chroma backbone
/// operating directly in pixel space. Composition: <see cref="ChromaRadianceImagePatchifier"/> (Conv2d 3→3072, k16/s16,
/// replaces VAE + <c>img_in</c>) → <see cref="ChromaTransformer.ForwardCore"/> (19 double + 38 single blocks, byte-identical
/// weights to classic Chroma) → <see cref="ChromaRadianceNerfHead"/> (per-patch hypernetwork, replaces <c>final_layer</c>).
/// Predicts <b>x0</b> [B, 3, H, W] in [-1, 1]; the pipeline converts to velocity via
/// <c>HartsyInference.Diffusion.Utilities.X0Prediction.ToVelocity</c>. No final modulation norm is applied before the
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

        // F16 block loop (HARTSY_DIT_F16): the backbone runs its residual stream at ChromaF16.ResidualDamp scale.
        // The img stream is damped into that regime at the ForwardCore F16 cast (Radiance has no x_embedder); the
        // backbone returns pre-head img tokens still at damp scale, so un-damp them exactly at the NeRF head's
        // param_generator GEMM alpha (img tokens feed only that Linear).
        if (DitDtype.Act == DType.F16)
            _nerfHead.UndampImgInput(1.0f / ChromaF16.ResidualDamp);
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

        // ForwardCore consumes (disposes) the token tensor and returns pre-final-norm img tokens. Radiance never
        // applies the final modulation rows — the NeRF head replaces final_layer entirely — so the modulation
        // table (now caller-built) is disposed right after the core.
        Tensor modTable = _backbone.BuildModTable(backend, timestep, (int)rgb.Shape[0]);
        Tensor imgOut = _backbone.ForwardCore(
            backend, tokens, encoderHidden, modTable, txtSeqLen, hPacked, wPacked, attentionMask);
        modTable.Dispose();

        Tensor x0 = _nerfHead.Forward(backend, rgb, imgOut);
        imgOut.Dispose();
        return x0;
    }

    /// <summary>Runs one full denoise step's transformer work: the cond pass and (for CFG) the uncond pass. The
    /// two passes share ONE modulation-table build (same timestep), ONE conv patchify of the noisy image, and ONE
    /// NeRF-head pixel embedding — each was previously recomputed per pass. Returned x0 tensors are caller-owned.</summary>
    /// <param name="rgb">Noisy pixel sample [B, 3, H, W]; must stay device/host consistent across the pair.</param>
    /// <param name="condContext">Conditional T5 embeddings [B, condLen, 4096].</param>
    /// <param name="uncondContext">Unconditional T5 embeddings, or null to skip the second pass.</param>
    public (Tensor condX0, Tensor? uncondX0) ForwardPaired(
        IBackend backend,
        Tensor rgb,
        Tensor condContext,
        Tensor? uncondContext,
        float timestep,
        Tensor? condMask,
        Tensor? uncondMask)
    {
        int batch = (int)rgb.Shape[0];
        int height = (int)rgb.Shape[2];
        int width = (int)rgb.Shape[3];
        int patch = _patchifier.PatchSize;
        int hPacked = height / patch;
        int wPacked = width / patch;
        int condTxtLen = (int)condContext.Shape[1];

        Tensor modTable = _backbone.BuildModTable(backend, timestep, batch);
        Tensor embed = _nerfHead.EmbedPixels(backend, rgb);
        Tensor tokens = _patchifier.Forward(backend, rgb);

        // ForwardCore consumes its token input — device-copy for the second pass before the first runs.
        Tensor? tokensUncond = null;
        if (uncondContext is not null)
        {
            tokensUncond = new Tensor(tokens.Shape, tokens.DType);
            backend.CopyInto(tokensUncond, tokens);
        }

        Tensor imgOutCond = _backbone.ForwardCore(
            backend, tokens, condContext, modTable, condTxtLen, hPacked, wPacked, condMask);
        Tensor condX0 = _nerfHead.ForwardFromEmbed(backend, embed, imgOutCond, rgb.Shape);
        imgOutCond.Dispose();

        Tensor? uncondX0 = null;
        if (uncondContext is not null)
        {
            int uncondTxtLen = (int)uncondContext.Shape[1];
            Tensor imgOutUncond = _backbone.ForwardCore(
                backend, tokensUncond!, uncondContext, modTable, uncondTxtLen, hPacked, wPacked, uncondMask);
            uncondX0 = _nerfHead.ForwardFromEmbed(backend, embed, imgOutUncond, rgb.Shape);
            imgOutUncond.Dispose();
        }

        embed.Dispose();
        modTable.Dispose();
        return (condX0, uncondX0);
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
