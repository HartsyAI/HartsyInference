using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Prompting;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Z-Image diffusion transformer (Lumina2/NextDiT). Processes Qwen3-encoded caption tokens through context_refiner, packed image latent tokens through noise_refiner with timestep modulation, then concatenates and runs through 30 main layers with multi-axis RoPE. The final layer applies AdaLN(scale, shift) to image tokens only and projects back to patch space.</summary>
public sealed unsafe class ZImageTransformer : IDisposable
{
    private readonly ZImageConfig _config;
    private readonly ZImageContextRefinerBlock[] _contextRefiners;
    private readonly ZImageBlock[] _noiseRefiners;
    private readonly ZImageBlock[] _layers;
    private readonly ZImageRope _rope;
    private int _disposed;

    // ── Per-generation caches (the Krea2 pattern) ──────────────────────────────────────────────────────────
    // The caption path (cap_embedder → pad → context_refiners) and all three RoPE precomputes are
    // timestep-INDEPENDENT, yet were recomputed every denoise step (~14 wasted block-forwards + 3 host
    // cos/sin table builds per step). Cache the refined caption keyed on the encoder-output reference (a new
    // prompt is a new tensor → evicts) and the rope tables keyed on shape signatures. Regional conditioning
    // bypasses the caption cache (its caption stream is plan/step-dependent).
    private Tensor? _cachedRefinedCaption;
    private object? _cachedCaptionKey;
    private int _cachedCapPaddedLen = -1;
    private ZImageRope? _captionRope;
    private ZImageRope? _refinerRope;
    private long _refinerRopeSig = long.MinValue;
    private long _fullRopeSig = long.MinValue;

    // t_embedder: sinusoidal(timestep × 1000) → Linear(adaLNDim → adaLNDim) → SiLU → Linear(adaLNDim → adaLNDim)
    private Tensor? _tEmbLinear1Weight, _tEmbLinear1Bias;
    private Tensor? _tEmbLinear2Weight, _tEmbLinear2Bias;

    // cap_embedder: RMSNorm(2560) → Linear(2560 → 3840). Only the RMSNorm scale is stored; cap_embedder.0 has no bias.
    private Tensor? _capEmbedderNormWeight;
    private Tensor? _capEmbedderLinearWeight, _capEmbedderLinearBias;

    // x_embedder: Linear(in_channels * patch² * f_patch → hidden). Z-Image patches as 16 * 2 * 2 * 1 = 64 → 3840. Tongyi single-file uses bare "x_embedder.{weight,bias}" (no patch suffix).
    private Tensor? _xEmbedderWeight, _xEmbedderBias;

    // final_layer: Sequential(SiLU, Linear(adaLNDim → hidden)) for scale-only AdaLN + Linear(hidden → out_channels * patch² * f_patch). The modulation outputs ONE chunk (just scale) — Z-Image's final layer omits the shift term, so the formula is `norm(x) * (1 + scale)` then linear.
    private Tensor? _finalAdaLNWeight, _finalAdaLNBias;
    private Tensor? _finalLinearWeight, _finalLinearBias;

    // Learned padding embeddings: cap_pad_token, x_pad_token, both shape [1, hidden].
    private Tensor? _capPadToken;
    private Tensor? _xPadToken;

    public ZImageTransformer(ZImageConfig config)
    {
        _config = config;

        _contextRefiners = new ZImageContextRefinerBlock[config.NumRefinerLayers];
        for (int i = 0; i < config.NumRefinerLayers; i++)
        {
            _contextRefiners[i] = new ZImageContextRefinerBlock(
                config.HiddenSize, config.NumHeads, config.FfnDim, config.NormEps);
        }

        _noiseRefiners = new ZImageBlock[config.NumRefinerLayers];
        for (int i = 0; i < config.NumRefinerLayers; i++)
        {
            _noiseRefiners[i] = new ZImageBlock(
                config.HiddenSize, config.NumHeads, config.FfnDim, config.NormEps);
        }

        _layers = new ZImageBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            _layers[i] = new ZImageBlock(
                config.HiddenSize, config.NumHeads, config.FfnDim, config.NormEps);
        }

        _rope = new ZImageRope(config.AxesDims, config.RopeTheta);
    }

    /// <summary>Loads all weights from a Z-Image diffusers/single-file dict (post-converter).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
        => LoadWeightsInternal(weights, requireFinalLayer: true);

    /// <summary>Weight-loading core shared with <see cref="ZetaChromaTransformer"/>. Zeta-Chroma checkpoints have
    /// no <c>final_layer.*</c> (replaced by the <c>dec_net.*</c> pixel decoder head), so those lookups become
    /// optional when <paramref name="requireFinalLayer"/> is false.</summary>
    internal void LoadWeightsInternal(IReadOnlyDictionary<string, Tensor> weights, bool requireFinalLayer)
    {
        _tEmbLinear1Weight = weights["t_embedder.mlp.0.weight"];
        weights.TryGetValue("t_embedder.mlp.0.bias", out _tEmbLinear1Bias);
        _tEmbLinear2Weight = weights["t_embedder.mlp.2.weight"];
        weights.TryGetValue("t_embedder.mlp.2.bias", out _tEmbLinear2Bias);

        // CudaBackend.RmsNorm reads the weight pointer as float* directly, so the scale tensor MUST be F32
        // (BF16/F16 norms would be reinterpreted as garbage F32 values). Cap_embedder.0 is the [2560]
        // RMSNorm scale — cheap to cast, mandatory for correctness.
        Tensor capNorm = weights["cap_embedder.0.weight"];
        _capEmbedderNormWeight = capNorm.DType == DType.F32 ? capNorm : capNorm.CastTo(DType.F32);
        _capEmbedderLinearWeight = weights["cap_embedder.1.weight"];
        weights.TryGetValue("cap_embedder.1.bias", out _capEmbedderLinearBias);

        _xEmbedderWeight = weights["x_embedder.weight"];
        weights.TryGetValue("x_embedder.bias", out _xEmbedderBias);

        if (requireFinalLayer)
        {
            _finalAdaLNWeight = weights["final_layer.adaLN_modulation.1.weight"];
            weights.TryGetValue("final_layer.adaLN_modulation.1.bias", out _finalAdaLNBias);
            _finalLinearWeight = weights["final_layer.linear.weight"];
            weights.TryGetValue("final_layer.linear.bias", out _finalLinearBias);
        }
        else
        {
            weights.TryGetValue("final_layer.adaLN_modulation.1.weight", out _finalAdaLNWeight);
            weights.TryGetValue("final_layer.adaLN_modulation.1.bias", out _finalAdaLNBias);
            weights.TryGetValue("final_layer.linear.weight", out _finalLinearWeight);
            weights.TryGetValue("final_layer.linear.bias", out _finalLinearBias);
        }

        // Pad-token raw float* access happens in PadCaption/PadImage, so force F32 here. These are
        // tiny ([1, 3840]) so the cast cost is negligible regardless of the checkpoint's native dtype.
        if (weights.TryGetValue("cap_pad_token", out Tensor? capPad))
            _capPadToken = capPad.DType == DType.F32 ? capPad : capPad.CastTo(DType.F32);
        if (weights.TryGetValue("x_pad_token", out Tensor? xPad))
            _xPadToken = xPad.DType == DType.F32 ? xPad : xPad.CastTo(DType.F32);

        for (int i = 0; i < _contextRefiners.Length; i++)
            _contextRefiners[i].LoadWeights(weights, $"context_refiner.{i}");
        for (int i = 0; i < _noiseRefiners.Length; i++)
            _noiseRefiners[i].LoadWeights(weights, $"noise_refiner.{i}");
        for (int i = 0; i < _layers.Length; i++)
            _layers[i].LoadWeights(weights, $"layers.{i}");
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_tEmbLinear1Weight is not null) yield return _tEmbLinear1Weight;
        if (_tEmbLinear1Bias is not null) yield return _tEmbLinear1Bias;
        if (_tEmbLinear2Weight is not null) yield return _tEmbLinear2Weight;
        if (_tEmbLinear2Bias is not null) yield return _tEmbLinear2Bias;
        if (_capEmbedderNormWeight is not null) yield return _capEmbedderNormWeight;
        if (_capEmbedderLinearWeight is not null) yield return _capEmbedderLinearWeight;
        if (_capEmbedderLinearBias is not null) yield return _capEmbedderLinearBias;
        if (_xEmbedderWeight is not null) yield return _xEmbedderWeight;
        if (_xEmbedderBias is not null) yield return _xEmbedderBias;
        if (_finalAdaLNWeight is not null) yield return _finalAdaLNWeight;
        if (_finalAdaLNBias is not null) yield return _finalAdaLNBias;
        if (_finalLinearWeight is not null) yield return _finalLinearWeight;
        if (_finalLinearBias is not null) yield return _finalLinearBias;
        if (_capPadToken is not null) yield return _capPadToken;
        if (_xPadToken is not null) yield return _xPadToken;

        for (int i = 0; i < _contextRefiners.Length; i++)
            foreach (Tensor w in _contextRefiners[i].EnumerateWeights()) yield return w;
        for (int i = 0; i < _noiseRefiners.Length; i++)
            foreach (Tensor w in _noiseRefiners[i].EnumerateWeights()) yield return w;
        for (int i = 0; i < _layers.Length; i++)
            foreach (Tensor w in _layers[i].EnumerateWeights()) yield return w;
    }

    /// <summary>Forward pass: predicts velocity for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="latent">Input latent [B, in_channels, H, W] in latent space (already VAE-scaled).</param>
    /// <param name="captionEmbeddings">Qwen3-encoded caption [B, capRealLen, capFeatDim=2560].</param>
    /// <param name="sigma">Current sigma (flow-match noise level, 0..1).</param>
    /// <returns>Predicted velocity [B, in_channels, H, W] (in patch-space → unpatchified).</returns>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor captionEmbeddings, float sigma,
        RegionalPlan? regionalPlan = null, int regionalStep = 0)
    {
        int batch = (int)latent.Shape[0];
        int inChannels = (int)latent.Shape[1];
        int patch = _config.PatchSize;

        (Tensor imgTokens, Tensor tEmb, int hPacked, int wPacked) =
            ForwardCore(backend, latent, captionEmbeddings, sigma, regionalPlan, regionalStep);
        int imgRealLen = hPacked * wPacked;

        // ── 8. Final layer: AdaLN(scale only) → LayerNorm-no-affine + modulate → Linear → unpatchify ──
        // Applied AFTER the pad-token trim — the final layer is per-token, so trimming first is equivalent
        // to the diffusers order (final layer on the padded sequence, then trim) and skips dead tokens.
        Tensor finalProj = ApplyFinalLayer(backend, imgTokens, tEmb, batch, imgRealLen);
        ZImageDebugDump.Dump("final_layer", finalProj);
        imgTokens.Dispose();
        tEmb.Dispose();

        Tensor velocity = Unpatchify(finalProj, batch, inChannels, hPacked, wPacked, patch);
        finalProj.Dispose();

        ZImageDebugDump.DumpOutput(velocity);
        return velocity;
    }

    /// <summary>Backbone forward shared with <see cref="ZetaChromaTransformer"/>: caption/noise refiners + the 30
    /// main layers, from raw latent to post-backbone image tokens (pad tokens already trimmed, final layer NOT
    /// applied). The caller owns both returned tensors; <c>tEmb</c> is returned so classic Z-Image can run its
    /// AdaLN final layer — Zeta-Chroma disposes it unused (its decoder head conditions on the tokens only).</summary>
    internal (Tensor imgTokens, Tensor tEmb, int hPacked, int wPacked) ForwardCore(
        IBackend backend, Tensor latent, Tensor captionEmbeddings, float sigma,
        RegionalPlan? regionalPlan = null, int regionalStep = 0)
    {
        int batch = (int)latent.Shape[0];
        int inChannels = (int)latent.Shape[1];
        int latentH = (int)latent.Shape[2];
        int latentW = (int)latent.Shape[3];
        int hidden = _config.HiddenSize;
        int patch = _config.PatchSize;

        if (latentH % patch != 0 || latentW % patch != 0)
            throw new ArgumentException($"Latent H/W ({latentH}x{latentW}) must be divisible by patch size {patch}.");

        int hPacked = latentH / patch;
        int wPacked = latentW / patch;
        int imgRealLen = hPacked * wPacked;
        int imgPaddedLen = PadUpTo(imgRealLen, _config.SeqMultiOf);

        // Regional conditioning: append region caption streams (so they share cap_embedder +
        // context_refiner) and remember each region's caption-relative column range + image-grid
        // mask. The [image|caption] attention bias is built below once padded lengths are known.
        Tensor effCaption = captionEmbeddings;
        bool ownsCaption = false;
        List<(int Start, int End)>? regionCapRanges = null;
        List<float[]>? regionGridMasks = null;
        if (regionalPlan is not null && regionalPlan.Regions.Count > 0)
        {
            (effCaption, _, regionCapRanges, regionGridMasks) =
                RegionalConditioningLayout.BuildTextStream(regionalPlan, captionEmbeddings, hPacked, wPacked);
            ownsCaption = true;
        }

        int capRealLen = (int)effCaption.Shape[1];
        int capPaddedLen = PadUpTo(capRealLen, _config.SeqMultiOf);

        // ── 1. Timestep embedding ──
        Tensor tEmb = ComputeTimestepEmbedding(backend, sigma, batch);
        ZImageDebugDump.Dump("t_embedder", tEmb);

        // ── 2. Caption embedding: cap_embedder + pad + context_refiner stack — CACHED across steps ──
        // Timestep-independent: same captionEmbeddings in → same refinedCaption out, so recomputing it every
        // step wasted 2 refiner block-forwards + the cap_embedder + a host rope build per step. Keyed on the
        // encoder-output reference + padded length; regional runs bypass (their caption stream is step-dependent).
        bool captionCacheable = regionalPlan is null || regionalPlan.Regions.Count == 0;
        bool captionCached = captionCacheable && ReferenceEquals(_cachedCaptionKey, captionEmbeddings)
            && _cachedCapPaddedLen == capPaddedLen && _cachedRefinedCaption is not null;
        Tensor refinedCaption;
        if (captionCached)
        {
            refinedCaption = _cachedRefinedCaption!;
        }
        else
        {
            Tensor capProjected = EmbedCaption(backend, effCaption, batch, capRealLen);
            if (ownsCaption)
            {
                effCaption.Dispose();
            }
            ZImageDebugDump.Dump("cap_embedder", capProjected);
            Tensor capPadded = PadCaption(capProjected, capRealLen, capPaddedLen, batch);
            // PadCaption may return the input unchanged when paddedLen==realLen — only dispose if it allocated a new tensor.
            if (!ReferenceEquals(capPadded, capProjected))
                capProjected.Dispose();

            // Caption-only RoPE: positions (1..capPaddedLen, 0, 0) on the frame axis. Diffusers ZImageTransformerBlock
            // applies freqs_cis even when modulation=False (the context_refiner case), so caption tokens DO get RoPE.
            _captionRope ??= new ZImageRope(_config.AxesDims, _config.RopeTheta);
            Tensor capPosIds = ZImageRope.BuildCaptionPositionIds(capPaddedLen);
            _captionRope.Precompute(capPosIds);
            capPosIds.Dispose();

            refinedCaption = capPadded;
            for (int i = 0; i < _contextRefiners.Length; i++)
            {
                Tensor next = _contextRefiners[i].Forward(backend, refinedCaption, _captionRope);
                refinedCaption.Dispose();
                refinedCaption = next;
                ZImageDebugDump.Dump($"context_refiner.{i}", refinedCaption);
            }

            if (captionCacheable)
            {
                _ = refinedCaption.DataPointer;   // materialize to host so it survives across steps
                _cachedRefinedCaption?.Dispose();
                _cachedRefinedCaption = refinedCaption;
                _cachedCaptionKey = captionEmbeddings;
                _cachedCapPaddedLen = capPaddedLen;
            }
        }
        bool ownsRefinedCaption = !captionCacheable;   // cached (or newly cached) captions are field-owned

        // ── 3. Image patchify + embed + pad + noise_refiner stack ──
        Tensor packedLatent = Patchify(latent, batch, inChannels, latentH, latentW, patch);
        TensorShape imgEmbShape = new TensorShape(batch, imgRealLen, hidden);
        Tensor imgEmbedded = new Tensor(imgEmbShape, latent.DType);
        backend.Linear(imgEmbedded, packedLatent, _xEmbedderWeight!, _xEmbedderBias);
        ZImageDebugDump.Dump("x_embedder", imgEmbedded);
        packedLatent.Dispose();

        Tensor imgPadded = PadImage(imgEmbedded, imgRealLen, imgPaddedLen, batch);
        if (!ReferenceEquals(imgPadded, imgEmbedded))
            imgEmbedded.Dispose();

        // Image-only RoPE for noise_refiner (cached by shape signature — timestep-independent).
        long refinerSig = ((long)hPacked * 19349663L) ^ ((long)wPacked * 83492791L) ^ ((long)imgPaddedLen * 2654435761L);
        _refinerRope ??= new ZImageRope(_config.AxesDims, _config.RopeTheta);
        if (_refinerRopeSig != refinerSig)
        {
            Tensor refinerPosIds = ZImageRope.BuildImagePositionIds(hPacked, wPacked, imgPaddedLen);
            _refinerRope.Precompute(refinerPosIds);
            refinerPosIds.Dispose();
            _refinerRopeSig = refinerSig;
        }

        Tensor refinedImage = imgPadded;
        for (int i = 0; i < _noiseRefiners.Length; i++)
        {
            Tensor next = _noiseRefiners[i].Forward(backend, refinedImage, tEmb, _refinerRope);
            refinedImage.Dispose();
            refinedImage = next;
            ZImageDebugDump.Dump($"noise_refiner.{i}", refinedImage);
        }

        // ── 4. Concatenate [refined_image, refined_caption] along sequence dim ──
        // Order is [image, caption] per diffusers transformer_z_image.py:859 — NOT [caption, image].
        // Device concat (backend.Concat = stream-ordered DtoD): the host ConcatAlongSeqDim read both inputs'
        // DataPointer every step — a full pipeline drain + D2H/H2D round-trip of the joint sequence.
        Tensor concat;
        if (batch == 1)
        {
            concat = new Tensor(new TensorShape(batch, imgPaddedLen + capPaddedLen, hidden), DType.F32);
            backend.Concat(concat, new[] { refinedImage, refinedCaption }, dim: 1);
        }
        else
        {
            concat = ConcatAlongSeqDim(refinedImage, refinedCaption, batch, imgPaddedLen, capPaddedLen, hidden);
        }
        if (ownsRefinedCaption)
            refinedCaption.Dispose();
        refinedImage.Dispose();

        // ── 5. Build full RoPE for the concatenated [image, caption] sequence (cached by signature) ──
        long fullSig = refinerSig ^ ((long)capPaddedLen * 73856093L);
        if (_fullRopeSig != fullSig)
        {
            Tensor fullPosIds = ZImageRope.BuildPositionIds(capPaddedLen, hPacked, wPacked, imgPaddedLen);
            _rope.Precompute(fullPosIds);
            fullPosIds.Dispose();
            _fullRopeSig = fullSig;
        }

        // Build the regional attention bias for the [image|caption] main-layer sequence. Image
        // tokens occupy [0, imgRealLen) (padded image tokens get no bias); region caption columns
        // are offset past the padded image block.
        Tensor? attnBias = null;
        if (regionCapRanges is not null)
        {
            float[] regionWeights = new float[regionalPlan!.Regions.Count];
            regionalPlan.ResolveStep(regionalStep, regionWeights);
            List<(int Start, int End)> absRanges = new List<(int Start, int End)>(regionCapRanges.Count);
            foreach ((int Start, int End) range in regionCapRanges)
            {
                absRanges.Add((imgPaddedLen + range.Start, imgPaddedLen + range.End));
            }
            attnBias = RegionalAttentionBias.Build(imgPaddedLen + capPaddedLen, 0, imgRealLen, absRanges, regionGridMasks!, regionWeights);
        }

        // ── 6. Main layers ──
        Tensor x = concat;
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, x, tEmb, _rope, attnBias);
            x.Dispose();
            x = next;
            ZImageDebugDump.Dump($"layers.{i}", x);
        }
        attnBias?.Dispose();

        // ── 7. Slice off image portion (front of the [image, caption] sequence), trim pad tokens ──
        // B=1: the real image tokens are the CONTIGUOUS front rows [0, imgRealLen) of x, so slice + pad-trim
        // collapse to ONE device SliceRows (the host helpers read x.DataPointer — a per-step pipeline drain).
        Tensor realImage;
        if (batch == 1)
        {
            realImage = new Tensor(new TensorShape(1, imgRealLen, hidden), DType.F32);
            backend.SliceRows(realImage, x, 0);
            x.Dispose();
        }
        else
        {
            Tensor imageSlice = SliceImageFront(x, batch, imgPaddedLen, capPaddedLen, hidden);
            x.Dispose();
            realImage = TrimImagePad(imageSlice, batch, imgRealLen, imgPaddedLen);
            if (!ReferenceEquals(realImage, imageSlice))
                imageSlice.Dispose();
        }

        return (realImage, tEmb, hPacked, wPacked);
    }

    /// <summary>Computes the timestep embedding [B, adaLNEmbedDim] via sinusoidal × 1000 → Linear → SiLU → Linear.</summary>
    private Tensor ComputeTimestepEmbedding(IBackend backend, float sigma, int batch)
    {
        int adaLNDim = _config.AdaLNEmbedDim;
        float scaled = sigma * _config.TimestepScale;

        TensorShape sinShape = new TensorShape(batch, adaLNDim);
        Tensor sinEmb = new Tensor(sinShape, DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinEmb, scaled, batch, adaLNDim);

        // mlp.0 projects sinusoidal[adaLNDim] → mlp_hidden (1024 in Z-Image-Turbo). Read the dim from the actual weight shape.
        int mlpHidden = (int)_tEmbLinear1Weight!.Shape[0];
        TensorShape midShape = new TensorShape(batch, mlpHidden);
        Tensor m1 = new Tensor(midShape, DType.F32);
        backend.Linear(m1, sinEmb, _tEmbLinear1Weight!, _tEmbLinear1Bias);
        sinEmb.Dispose();

        Tensor m1Act = new Tensor(midShape, DType.F32);
        backend.Silu(m1Act, m1);
        m1.Dispose();

        // mlp.2 projects mlp_hidden → adaLNDim (output back to the AdaLN-feeding dimension).
        TensorShape outShape = new TensorShape(batch, adaLNDim);
        Tensor tEmb = new Tensor(outShape, DType.F32);
        backend.Linear(tEmb, m1Act, _tEmbLinear2Weight!, _tEmbLinear2Bias);
        m1Act.Dispose();

        return tEmb;
    }

    /// <summary>Caption embedding: RMSNorm(cap_embedder.0) → Linear(cap_embedder.1) → [B, capLen, hidden].</summary>
    private Tensor EmbedCaption(IBackend backend, Tensor captionEmb, int batch, int capLen)
    {
        int capFeatDim = _config.CapFeatDim;
        int hidden = _config.HiddenSize;

        TensorShape inShape = new TensorShape(batch, capLen, capFeatDim);
        Tensor normed = new Tensor(inShape, captionEmb.DType);
        backend.RmsNorm(normed, captionEmb, _capEmbedderNormWeight!, _config.NormEps);

        TensorShape outShape = new TensorShape(batch, capLen, hidden);
        Tensor projected = new Tensor(outShape, captionEmb.DType);
        backend.Linear(projected, normed, _capEmbedderLinearWeight!, _capEmbedderLinearBias);
        normed.Dispose();

        return projected;
    }

    /// <summary>Pads caption tokens up to <paramref name="paddedLen"/> using <c>cap_pad_token</c> for the trailing slots.</summary>
    private Tensor PadCaption(Tensor capProjected, int realLen, int paddedLen, int batch)
    {
        if (paddedLen == realLen)
            return capProjected;

        int hidden = _config.HiddenSize;
        TensorShape outShape = new TensorShape(batch, paddedLen, hidden);
        Tensor output = new Tensor(outShape, capProjected.DType);

        float* srcPtr = (float*)capProjected.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        float* padPtr = _capPadToken is not null ? (float*)_capPadToken.DataPointer : null;

        for (int b = 0; b < batch; b++)
        {
            // Copy real tokens
            long realBytes = (long)realLen * hidden * sizeof(float);
            Buffer.MemoryCopy(srcPtr + b * realLen * hidden, outPtr + b * paddedLen * hidden, realBytes, realBytes);

            // Fill padding slots with cap_pad_token (broadcast) or zero
            for (int s = realLen; s < paddedLen; s++)
            {
                int slotOffset = (b * paddedLen + s) * hidden;
                if (padPtr != null)
                {
                    Buffer.MemoryCopy(padPtr, outPtr + slotOffset, hidden * sizeof(float), hidden * sizeof(float));
                }
                else
                {
                    for (int d = 0; d < hidden; d++)
                        outPtr[slotOffset + d] = 0f;
                }
            }
        }

        return output;
    }

    /// <summary>Pads image tokens up to <paramref name="paddedLen"/> using <c>x_pad_token</c>.</summary>
    private Tensor PadImage(Tensor imgEmbedded, int realLen, int paddedLen, int batch)
    {
        if (paddedLen == realLen)
            return imgEmbedded;

        int hidden = _config.HiddenSize;
        TensorShape outShape = new TensorShape(batch, paddedLen, hidden);
        Tensor output = new Tensor(outShape, imgEmbedded.DType);

        float* srcPtr = (float*)imgEmbedded.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        float* padPtr = _xPadToken is not null ? (float*)_xPadToken.DataPointer : null;

        for (int b = 0; b < batch; b++)
        {
            long realBytes = (long)realLen * hidden * sizeof(float);
            Buffer.MemoryCopy(srcPtr + b * realLen * hidden, outPtr + b * paddedLen * hidden, realBytes, realBytes);

            for (int s = realLen; s < paddedLen; s++)
            {
                int slotOffset = (b * paddedLen + s) * hidden;
                if (padPtr != null)
                {
                    Buffer.MemoryCopy(padPtr, outPtr + slotOffset, hidden * sizeof(float), hidden * sizeof(float));
                }
                else
                {
                    for (int d = 0; d < hidden; d++)
                        outPtr[slotOffset + d] = 0f;
                }
            }
        }

        return output;
    }

    /// <summary>Patchify: [B, C, H, W] → [B, hPacked*wPacked, pH*pW*C]. Diffusers Z-Image (`transformer_z_image.py:542-549`) uses inner-most order (pH, pW, C) — channel is the FASTEST-VARYING axis inside each patch, NOT the slowest. The matching <c>x_embedder</c> Linear was trained with this convention. </summary>
    internal static Tensor Patchify(Tensor latent, int batch, int channels, int H, int W, int patch)
    {
        int hPacked = H / patch;
        int wPacked = W / patch;
        int seqLen = hPacked * wPacked;
        int patchDim = patch * patch * channels;

        TensorShape outShape = new TensorShape(batch, seqLen, patchDim);
        Tensor output = new Tensor(outShape, latent.DType);

        float* inPtr = (float*)latent.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int hp = 0; hp < hPacked; hp++)
            {
                for (int wp = 0; wp < wPacked; wp++)
                {
                    int seqIdx = hp * wPacked + wp;
                    int outOffset = (b * seqLen + seqIdx) * patchDim;

                    // Inner ordering (slowest-to-fastest): pH, pW, C.
                    int outIdx = 0;
                    for (int ph = 0; ph < patch; ph++)
                    {
                        for (int pw = 0; pw < patch; pw++)
                        {
                            for (int c = 0; c < channels; c++)
                            {
                                int srcH = hp * patch + ph;
                                int srcW = wp * patch + pw;
                                int srcOffset = ((b * channels + c) * H + srcH) * W + srcW;
                                outPtr[outOffset + outIdx++] = inPtr[srcOffset];
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    /// <summary>Inverse of Patchify: [B, hPacked*wPacked, pH*pW*C] → [B, C, H, W]. Inner ordering matches Patchify (pH, pW, C).</summary>
    internal static Tensor Unpatchify(Tensor packed, int batch, int channels, int hPacked, int wPacked, int patch)
    {
        int H = hPacked * patch;
        int W = wPacked * patch;
        int seqLen = hPacked * wPacked;
        int patchDim = patch * patch * channels;

        TensorShape outShape = new TensorShape(batch, channels, H, W);
        Tensor output = new Tensor(outShape, packed.DType);

        float* inPtr = (float*)packed.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int hp = 0; hp < hPacked; hp++)
            {
                for (int wp = 0; wp < wPacked; wp++)
                {
                    int seqIdx = hp * wPacked + wp;
                    int inOffset = (b * seqLen + seqIdx) * patchDim;

                    int inIdx = 0;
                    for (int ph = 0; ph < patch; ph++)
                    {
                        for (int pw = 0; pw < patch; pw++)
                        {
                            for (int c = 0; c < channels; c++)
                            {
                                int dstH = hp * patch + ph;
                                int dstW = wp * patch + pw;
                                int dstOffset = ((b * channels + c) * H + dstH) * W + dstW;
                                outPtr[dstOffset] = inPtr[inOffset + inIdx++];
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    /// <summary>Concatenates two [B, S1, D] and [B, S2, D] tensors along the sequence dimension.</summary>
    private static Tensor ConcatAlongSeqDim(Tensor a, Tensor b, int batch, int seqA, int seqB, int dim)
    {
        int totalSeq = seqA + seqB;
        TensorShape outShape = new TensorShape(batch, totalSeq, dim);
        Tensor output = new Tensor(outShape, a.DType);

        float* aPtr = (float*)a.DataPointer;
        float* bPtr = (float*)b.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int bi = 0; bi < batch; bi++)
        {
            long aBytes = (long)seqA * dim * sizeof(float);
            long bBytes = (long)seqB * dim * sizeof(float);
            Buffer.MemoryCopy(aPtr + bi * seqA * dim, outPtr + bi * totalSeq * dim, aBytes, aBytes);
            Buffer.MemoryCopy(bPtr + bi * seqB * dim, outPtr + bi * totalSeq * dim + seqA * dim, bBytes, bBytes);
        }
        return output;
    }

    /// <summary>Slices the image-token portion off the front of the [image, caption] concatenated sequence: [B, imgLen+capLen, D] → [B, imgLen, D].</summary>
    private static Tensor SliceImageFront(Tensor combined, int batch, int imgLen, int capLen, int dim)
    {
        int totalSeq = imgLen + capLen;
        TensorShape outShape = new TensorShape(batch, imgLen, dim);
        Tensor output = new Tensor(outShape, combined.DType);

        float* srcPtr = (float*)combined.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            long bytes = (long)imgLen * dim * sizeof(float);
            Buffer.MemoryCopy(srcPtr + b * totalSeq * dim, outPtr + b * imgLen * dim, bytes, bytes);
        }
        return output;
    }

    /// <summary>Trims padded image slots off the back of the sequence: [B, paddedLen, D] → [B, realLen, D].</summary>
    private static Tensor TrimImagePad(Tensor padded, int batch, int realLen, int paddedLen)
    {
        int dim = (int)padded.Shape[2];
        if (realLen == paddedLen)
            return padded;

        TensorShape outShape = new TensorShape(batch, realLen, dim);
        Tensor output = new Tensor(outShape, padded.DType);

        float* srcPtr = (float*)padded.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            long bytes = (long)realLen * dim * sizeof(float);
            Buffer.MemoryCopy(srcPtr + b * paddedLen * dim, outPtr + b * realLen * dim, bytes, bytes);
        }
        return output;
    }

    /// <summary>Final layer: AdaLN(SiLU(t_emb)) → 1 chunk (scale only — Z-Image omits shift) → LayerNorm-no-affine + modulate → Linear projection back to patch space.</summary>
    private Tensor ApplyFinalLayer(IBackend backend, Tensor imageTokens, Tensor tEmb, int batch, int seqLen)
    {
        int hidden = _config.HiddenSize;
        int patchOutDim = _config.InChannels * _config.PatchSize * _config.PatchSize * _config.FramePatchSize;

        // SiLU first (final_layer's adaLN_modulation Sequential is `[SiLU, Linear]`).
        TensorShape tShape = new TensorShape(batch, _config.AdaLNEmbedDim);
        Tensor activated = new Tensor(tShape, tEmb.DType);
        backend.Silu(activated, tEmb);

        // Linear: adaLNEmbedDim → hidden (single chunk; scale only).
        TensorShape modShape = new TensorShape(batch, hidden);
        Tensor scaleParam = new Tensor(modShape, tEmb.DType);
        backend.Linear(scaleParam, activated, _finalAdaLNWeight!, _finalAdaLNBias);
        activated.Dispose();

        // LayerNorm-no-affine on image tokens. Diffusers Z-Image FinalLayer uses LayerNorm(eps=1e-6, elementwise_affine=False) — note: 1e-6, not the 1e-5 used elsewhere in the model.
        TensorShape seqShape = new TensorShape(batch, seqLen, hidden);
        Tensor normed = new Tensor(seqShape, imageTokens.DType);
        DiTUtils.LayerNormNoAffine(normed, imageTokens, batch, seqLen, hidden, 1e-6f);

        // Apply scale only: out = normed * (1 + scale).
        Tensor modulated = new Tensor(seqShape, imageTokens.DType);
        float* normPtr = (float*)normed.DataPointer;
        float* scalePtr = (float*)scaleParam.DataPointer;
        float* outPtr = (float*)modulated.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int condBase = b * hidden;
            for (int s = 0; s < seqLen; s++)
            {
                int seqOff = (b * seqLen + s) * hidden;
                for (int d = 0; d < hidden; d++)
                {
                    outPtr[seqOff + d] = normPtr[seqOff + d] * (1.0f + scalePtr[condBase + d]);
                }
            }
        }
        normed.Dispose();
        scaleParam.Dispose();

        // Linear projection to patch space
        TensorShape outShape = new TensorShape(batch, seqLen, patchOutDim);
        Tensor projected = new Tensor(outShape, imageTokens.DType);
        backend.Linear(projected, modulated, _finalLinearWeight!, _finalLinearBias);
        modulated.Dispose();

        return projected;
    }

    private static int PadUpTo(int n, int multiple)
    {
        int rem = n % multiple;
        return rem == 0 ? n : n + (multiple - rem);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _cachedRefinedCaption?.Dispose();
            _cachedRefinedCaption = null;
            _cachedCaptionKey = null;
            _tEmbLinear1Weight = null;
            _tEmbLinear1Bias = null;
            _tEmbLinear2Weight = null;
            _tEmbLinear2Bias = null;
            _capEmbedderNormWeight = null;
            _capEmbedderLinearWeight = null;
            _capEmbedderLinearBias = null;
            _xEmbedderWeight = null;
            _xEmbedderBias = null;
            _finalAdaLNWeight = null;
            _finalAdaLNBias = null;
            _finalLinearWeight = null;
            _finalLinearBias = null;
            _capPadToken = null;
            _xPadToken = null;
        }
        GC.SuppressFinalize(this);
    }
}
