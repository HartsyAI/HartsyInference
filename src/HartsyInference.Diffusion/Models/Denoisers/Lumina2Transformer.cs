using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Lumina-Image-2.0 diffusion transformer (NextDiT family, sibling of Z-Image). Processes Gemma-2-encoded caption tokens through a context_refiner stack, packed image latent tokens through a noise_refiner stack with timestep modulation, then concatenates as <c>[caption, image]</c> (captions FIRST — opposite of Z-Image's image-first concat) and runs through 26 main layers with multi-axis RoPE [32,32,32] θ=10000. The final layer applies <c>LuminaLayerNormContinuous</c> (a RMSNorm + scale-only modulation + projection) to image tokens only and projects back to patch space. <strong>Differences from Z-Image:</strong> separate Q/K/V (no fused QKV), GQA (24:8), no learned <c>cap_pad_token</c>/<c>x_pad_token</c>, no fixed-multiple-of padding, captions-first concat, larger AdaLN embed dim (1024 vs 256). See diffusers <c>transformer_lumina2.py</c>.</summary>
public sealed unsafe class Lumina2Transformer : IDisposable
{
    private readonly Lumina2Config _config;
    private readonly Lumina2ContextRefinerBlock[] _contextRefiners;

    // Per-forward-invariant caches (the 650s wall was dominated by re-doing all of this every forward):
    // refined captions are prompt-invariant (2-slot, keyed on the embedding tensor identity — the pipeline
    // passes the same cached cond/uncond tensors every step), and the three rope tables are geometry-keyed.
    private long _refinedCapKeyA = long.MinValue, _refinedCapKeyB = long.MinValue;
    private Tensor? _refinedCapA, _refinedCapB;
    private readonly ZImageRope _captionRopeCached;
    private readonly ZImageRope _refinerRopeCached;
    private int _capRopeSig = -1, _refinerRopeSig = -1, _jointRopeSig = -1;
    private readonly Lumina2Block[] _noiseRefiners;
    private readonly Lumina2Block[] _layers;
    private readonly ZImageRope _rope;
    private int _disposed;

    // time_caption_embed.timestep_embedder: sinusoidal(timestep) -> Linear(freqEmbDim -> adaLNDim) -> SiLU -> Linear(adaLNDim -> adaLNDim).
    // Diffusers names: time_caption_embed.timestep_embedder.linear_1.{weight,bias} and .linear_2.{weight,bias}.
    private Tensor? _tEmbLinear1Weight, _tEmbLinear1Bias;
    private Tensor? _tEmbLinear2Weight, _tEmbLinear2Bias;

    // time_caption_embed.caption_embedder: Sequential(RMSNorm(capFeatDim), Linear(capFeatDim -> hidden, bias=True)).
    // Diffusers names: time_caption_embed.caption_embedder.0.weight (RMSNorm scale)
    //                  time_caption_embed.caption_embedder.1.{weight,bias} (Linear).
    private Tensor? _capEmbedderNormWeight;
    private Tensor? _capEmbedderLinearWeight, _capEmbedderLinearBias;

    // x_embedder: Linear(in_channels * patch^2 -> hidden) with bias.
    private Tensor? _xEmbedderWeight, _xEmbedderBias;

    // norm_out: LuminaLayerNormContinuous(embed_dim=hidden, conditioning_dim=adaLNEmbedDim, out_dim=patch_dim).
    // Diffusers names: norm_out.linear_1.{weight,bias} (conditioning -> hidden scale)
    //                  norm_out.linear_2.{weight,bias} (hidden -> patch_dim output projection).
    private Tensor? _normOutLinear1Weight, _normOutLinear1Bias;
    private Tensor? _normOutLinear2Weight, _normOutLinear2Bias;

    public Lumina2Transformer(Lumina2Config config)
    {
        _config = config;

        _contextRefiners = new Lumina2ContextRefinerBlock[config.NumRefinerLayers];
        for (int i = 0; i < config.NumRefinerLayers; i++)
        {
            _contextRefiners[i] = new Lumina2ContextRefinerBlock(
                config.HiddenSize, config.NumHeads, config.NumKvHeads, config.HeadDim, config.FfnDim, config.NormEps);
        }

        _noiseRefiners = new Lumina2Block[config.NumRefinerLayers];
        for (int i = 0; i < config.NumRefinerLayers; i++)
        {
            _noiseRefiners[i] = new Lumina2Block(
                config.HiddenSize, config.NumHeads, config.NumKvHeads, config.HeadDim, config.FfnDim, config.AdaLNEmbedDim, config.NormEps);
        }

        _layers = new Lumina2Block[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            _layers[i] = new Lumina2Block(
                config.HiddenSize, config.NumHeads, config.NumKvHeads, config.HeadDim, config.FfnDim, config.AdaLNEmbedDim, config.NormEps);
        }

        _rope = new ZImageRope(config.AxesDims, config.RopeTheta);
        _captionRopeCached = new ZImageRope(config.AxesDims, config.RopeTheta);
        _refinerRopeCached = new ZImageRope(config.AxesDims, config.RopeTheta);
    }

    /// <summary>Loads all weights from a Lumina-Image-2.0 diffusers-format dict.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _tEmbLinear1Weight = weights["time_caption_embed.timestep_embedder.linear_1.weight"];
        weights.TryGetValue("time_caption_embed.timestep_embedder.linear_1.bias", out _tEmbLinear1Bias);
        _tEmbLinear2Weight = weights["time_caption_embed.timestep_embedder.linear_2.weight"];
        weights.TryGetValue("time_caption_embed.timestep_embedder.linear_2.bias", out _tEmbLinear2Bias);

        // The cap_embedder RMSNorm scale must be F32 (RmsNorm reads weight pointer as float* directly).
        Tensor capNorm = weights["time_caption_embed.caption_embedder.0.weight"];
        _capEmbedderNormWeight = capNorm.DType == DType.F32 ? capNorm : capNorm.CastTo(DType.F32);
        _capEmbedderLinearWeight = weights["time_caption_embed.caption_embedder.1.weight"];
        weights.TryGetValue("time_caption_embed.caption_embedder.1.bias", out _capEmbedderLinearBias);

        _xEmbedderWeight = weights["x_embedder.weight"];
        weights.TryGetValue("x_embedder.bias", out _xEmbedderBias);

        _normOutLinear1Weight = weights["norm_out.linear_1.weight"];
        weights.TryGetValue("norm_out.linear_1.bias", out _normOutLinear1Bias);
        _normOutLinear2Weight = weights["norm_out.linear_2.weight"];
        weights.TryGetValue("norm_out.linear_2.bias", out _normOutLinear2Bias);

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
        foreach (Tensor w in EnumerateSharedWeights()) yield return w;
        for (int i = 0; i < _layers.Length; i++)
            foreach (Tensor w in _layers[i].EnumerateWeights()) yield return w;
    }

    /// <summary>The number of streamable main <see cref="Lumina2Block"/>s (26). The context/noise refiner
    /// stacks are NOT included — they always run once as a prefix stage on the split's FIRST backend, same as
    /// Krea2's text-fusion stage.</summary>
    public int BlockCount => _layers.Length;

    /// <summary>The non-main-layer weights: timestep/caption embedders, x_embedder, norm_out, and BOTH refiner
    /// stacks (context_refiner + noise_refiner) — always resident on the block-range split's FIRST backend
    /// (see <see cref="ForwardSharded"/>), since they run once, before the shardable main-layer loop.</summary>
    public IEnumerable<Tensor> EnumerateSharedWeights()
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
        if (_normOutLinear1Weight is not null) yield return _normOutLinear1Weight;
        if (_normOutLinear1Bias is not null) yield return _normOutLinear1Bias;
        if (_normOutLinear2Weight is not null) yield return _normOutLinear2Weight;
        if (_normOutLinear2Bias is not null) yield return _normOutLinear2Bias;

        for (int i = 0; i < _contextRefiners.Length; i++)
            foreach (Tensor w in _contextRefiners[i].EnumerateWeights()) yield return w;
        for (int i = 0; i < _noiseRefiners.Length; i++)
            foreach (Tensor w in _noiseRefiners[i].EnumerateWeights()) yield return w;
    }

    /// <summary>Enumerates only main layers <c>[startBlock, endBlock)</c>'s weights — the asymmetric preload a
    /// block-range shard needs (see <see cref="ForwardSharded"/>).</summary>
    public IEnumerable<Tensor> EnumerateBlockRangeWeights(int startBlock, int endBlock)
    {
        for (int i = startBlock; i < endBlock; i++)
            foreach (Tensor w in _layers[i].EnumerateWeights()) yield return w;
    }

    /// <summary>Forward pass: predicts velocity for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="latent">Input latent [B, in_channels, H, W] in latent space (already VAE-scaled).</param>
    /// <param name="captionEmbeddings">Gemma-2-encoded caption [B, capLen, capFeatDim=2304].</param>
    /// <param name="sigma">Current sigma (flow-match noise level, typically 1 - t/num_train_timesteps as Lumina 2.0 inverts the schedule).</param>
    /// <returns>Predicted velocity [B, in_channels, H, W] (in patch-space → unpatchified).</returns>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor captionEmbeddings, float sigma)
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
        int imgLen = hPacked * wPacked;
        int capLen = (int)captionEmbeddings.Shape[1];

        // ── 1. Timestep embedding ──
        Tensor tEmb = ComputeTimestepEmbedding(backend, sigma, batch);
        Lumina2DebugDump.Dump("t_embedder", tEmb);

        // ── 2. Caption embedding + context_refiner — PROMPT-INVARIANT, so cached per embedding tensor
        //       (two slots: cond + uncond). The refiner stack previously re-ran every forward of every step. ──
        long capKey = ((long)System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(captionEmbeddings) << 16)
            ^ (uint)capLen;
        Tensor refinedCaption;
        if (capKey == _refinedCapKeyA)
        {
            refinedCaption = _refinedCapA!;
        }
        else if (capKey == _refinedCapKeyB)
        {
            refinedCaption = _refinedCapB!;
        }
        else
        {
            Tensor capProjected = EmbedCaption(backend, captionEmbeddings, batch, capLen);
            Lumina2DebugDump.Dump("cap_embedder", capProjected);

            // Caption-only RoPE: positions (0..capLen-1, 0, 0), pos_start=0 (diffusers Lumina2RotaryPosEmbed).
            if (_capRopeSig != capLen)
            {
                Tensor capPosIds = BuildCaptionPositionIds(capLen);
                _captionRopeCached.Precompute(capPosIds);
                capPosIds.Dispose();
                _capRopeSig = capLen;
            }

            refinedCaption = capProjected;
            for (int i = 0; i < _contextRefiners.Length; i++)
            {
                Tensor next = _contextRefiners[i].Forward(backend, refinedCaption, _captionRopeCached);
                refinedCaption.Dispose();
                refinedCaption = next;
                Lumina2DebugDump.Dump($"context_refiner.{i}", refinedCaption);
            }
            _ = refinedCaption.DataPointer;   // host-materialize so the cache survives activation sweeps

            // Install (B slot takes the old A entry, so cond+uncond alternate without churn).
            if (_refinedCapB != refinedCaption) _refinedCapB?.Dispose();
            _refinedCapB = _refinedCapA;
            _refinedCapKeyB = _refinedCapKeyA;
            _refinedCapA = refinedCaption;
            _refinedCapKeyA = capKey;
        }

        // ── 3. Image patchify + embed → noise_refiner stack with timestep modulation ──
        Tensor packedLatent = Patchify(latent, batch, inChannels, latentH, latentW, patch);
        TensorShape imgEmbShape = new TensorShape(batch, imgLen, hidden);
        Tensor imgEmbedded = new Tensor(imgEmbShape, latent.DType);
        backend.Linear(imgEmbedded, packedLatent, _xEmbedderWeight!, _xEmbedderBias);
        Lumina2DebugDump.Dump("x_embedder", imgEmbedded);
        packedLatent.Dispose();

        // Image-only RoPE for noise_refiner. Lumina 2.0 uses (0, row, col) for image tokens before
        // they get the caption-frame offset (which only matters in the joint sequence).
        int refinerSig = hPacked << 16 | wPacked;
        if (_refinerRopeSig != refinerSig)
        {
            Tensor refinerPosIds = BuildImagePositionIds(hPacked, wPacked, imageFrameStart: 0);
            _refinerRopeCached.Precompute(refinerPosIds);
            refinerPosIds.Dispose();
            _refinerRopeSig = refinerSig;
        }

        Tensor refinedImage = imgEmbedded;
        for (int i = 0; i < _noiseRefiners.Length; i++)
        {
            Tensor next = _noiseRefiners[i].Forward(backend, refinedImage, tEmb, _refinerRopeCached);
            refinedImage.Dispose();
            refinedImage = next;
            Lumina2DebugDump.Dump($"noise_refiner.{i}", refinedImage);
        }

        // ── 4. Concatenate [refined_caption, refined_image] along sequence dim ──
        // Lumina 2.0 puts captions FIRST (diffusers transformer_lumina2.py — joint_hidden_states with
        // captions concatenated before image patches). Z-Image's order is the opposite.
        Tensor concat;
        if (batch == 1)
        {
            concat = new Tensor(new TensorShape(1, capLen + imgLen, hidden), refinedImage.DType);
            backend.Concat(concat, new Tensor[] { refinedCaption, refinedImage }, 1);
        }
        else
        {
            concat = ConcatAlongSeqDim(refinedCaption, refinedImage, batch, capLen, imgLen, hidden);
        }
        refinedImage.Dispose();   // refinedCaption is cache-owned — not disposed here

        // ── 5. Build full RoPE for the concatenated [caption, image] sequence ──
        int jointSig = (capLen << 20) ^ (hPacked << 10) ^ wPacked;
        if (_jointRopeSig != jointSig)
        {
            Tensor fullPosIds = BuildJointPositionIds(capLen, hPacked, wPacked);
            _rope.Precompute(fullPosIds);
            fullPosIds.Dispose();
            _jointRopeSig = jointSig;
        }

        // ── 6. Main layers ──
        Tensor x = concat;
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, x, tEmb, _rope);
            x.Dispose();
            x = next;
            Lumina2DebugDump.Dump($"layers.{i}", x);
        }

        // ── 7. Slice off image portion (BACK of the [caption, image] sequence; B=1: device row slice) ──
        Tensor imageSlice;
        if (batch == 1)
        {
            imageSlice = new Tensor(new TensorShape(1, imgLen, hidden), x.DType);
            backend.SliceRows(imageSlice, x, capLen);
        }
        else
        {
            imageSlice = SliceImageBack(x, batch, capLen, imgLen, hidden);
        }
        x.Dispose();

        // ── 8. norm_out: LuminaLayerNormContinuous → unpatchify ──
        Tensor finalProj = ApplyNormOut(backend, imageSlice, tEmb, batch, imgLen);
        Lumina2DebugDump.Dump("norm_out", finalProj);
        imageSlice.Dispose();
        tEmb.Dispose();

        Tensor velocity;
        if (batch == 1)
        {
            velocity = new Tensor(new TensorShape(1, inChannels, latentH, latentW), DType.F32);
            backend.UnpatchifyTokens(velocity, finalProj, inChannels, hPacked, wPacked, patch, innerChannelFastest: true);
        }
        else
        {
            velocity = Unpatchify(finalProj, batch, inChannels, hPacked, wPacked, patch);
        }
        finalProj.Dispose();

        Lumina2DebugDump.DumpOutput(velocity);
        return velocity;
    }

    /// <summary>Predicts velocity for one denoising step with the main <see cref="Lumina2Block"/> loop split
    /// across two backends: layers <c>[0, splitBlock)</c> run on <paramref name="backendA"/>, layers
    /// <c>[splitBlock, BlockCount)</c> on <paramref name="backendB"/> — a sequential pipeline split, not
    /// tensor-parallel, so there is <b>no latency win</b>; the win is VRAM pooling. <paramref name="backendA"/>
    /// must hold <see cref="EnumerateSharedWeights"/> + <see cref="EnumerateBlockRangeWeights"/><c>(0,
    /// splitBlock)</c>; <paramref name="backendB"/> must hold ONLY <see cref="EnumerateBlockRangeWeights"/>
    /// <c>(splitBlock, BlockCount)</c>. The context/noise-refiner prefix always runs on
    /// <paramref name="backendA"/> (same "prefix stage" shape as Krea2's text-fusion) — unlike <see cref="Forward"/>,
    /// this does NOT use the prompt-invariant refined-caption cache (<c>_refinedCapA</c>/<c>_refinedCapB</c>):
    /// sharding already forgoes a latency win, so the cache's cross-step reuse isn't worth the extra bookkeeping
    /// of tracking cache ownership across two backends. Excludes <see cref="Forward"/>'s across-step caches
    /// entirely; use <see cref="Forward"/> for those.</summary>
    public Tensor ForwardSharded(IBackend backendA, IBackend backendB, Tensor latent, Tensor captionEmbeddings,
        float sigma, int splitBlock)
    {
        if (splitBlock <= 0 || splitBlock >= _layers.Length)
            throw new ArgumentOutOfRangeException(nameof(splitBlock),
                $"splitBlock must be in (0, {_layers.Length}) — 0 or {_layers.Length} would put the whole NextDiT on one backend.");

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
        int imgLen = hPacked * wPacked;
        int capLen = (int)captionEmbeddings.Shape[1];

        Tensor tEmb = ComputeTimestepEmbedding(backendA, sigma, batch);

        // ── caption embedding + context_refiner (always fresh — no prompt-invariant cache, see doc above) ──
        Tensor capProjected = EmbedCaption(backendA, captionEmbeddings, batch, capLen);
        if (_capRopeSig != capLen)
        {
            Tensor capPosIds = BuildCaptionPositionIds(capLen);
            _captionRopeCached.Precompute(capPosIds);
            capPosIds.Dispose();
            _capRopeSig = capLen;
        }
        Tensor refinedCaption = capProjected;
        for (int i = 0; i < _contextRefiners.Length; i++)
        {
            Tensor next = _contextRefiners[i].Forward(backendA, refinedCaption, _captionRopeCached);
            refinedCaption.Dispose();
            refinedCaption = next;
        }

        // ── image patchify + embed → noise_refiner stack with timestep modulation ──
        Tensor packedLatent = Patchify(latent, batch, inChannels, latentH, latentW, patch);
        Tensor imgEmbedded = new Tensor(new TensorShape(batch, imgLen, hidden), latent.DType);
        backendA.Linear(imgEmbedded, packedLatent, _xEmbedderWeight!, _xEmbedderBias);
        packedLatent.Dispose();

        int refinerSig = hPacked << 16 | wPacked;
        if (_refinerRopeSig != refinerSig)
        {
            Tensor refinerPosIds = BuildImagePositionIds(hPacked, wPacked, imageFrameStart: 0);
            _refinerRopeCached.Precompute(refinerPosIds);
            refinerPosIds.Dispose();
            _refinerRopeSig = refinerSig;
        }

        Tensor refinedImage = imgEmbedded;
        for (int i = 0; i < _noiseRefiners.Length; i++)
        {
            Tensor next = _noiseRefiners[i].Forward(backendA, refinedImage, tEmb, _refinerRopeCached);
            refinedImage.Dispose();
            refinedImage = next;
        }

        // ── concatenate [refined_caption, refined_image] (captions first, mirroring Forward) ──
        Tensor concat;
        if (batch == 1)
        {
            concat = new Tensor(new TensorShape(1, capLen + imgLen, hidden), refinedImage.DType);
            backendA.Concat(concat, new Tensor[] { refinedCaption, refinedImage }, 1);
        }
        else
        {
            concat = ConcatAlongSeqDim(refinedCaption, refinedImage, batch, capLen, imgLen, hidden);
        }
        refinedImage.Dispose();
        refinedCaption.Dispose();

        // ── joint RoPE for the concatenated [caption, image] sequence ──
        int jointSig = (capLen << 20) ^ (hPacked << 10) ^ wPacked;
        if (_jointRopeSig != jointSig)
        {
            Tensor fullPosIds = BuildJointPositionIds(capLen, hPacked, wPacked);
            _rope.Precompute(fullPosIds);
            fullPosIds.Dispose();
            _jointRopeSig = jointSig;
        }

        // ── main layers, split across the two backends ──
        Tensor xA = ForwardLayersRange(backendA, concat, tEmb, 0, splitBlock);

        // ── hand off to backendB: x + tEmb (every layer's AdaLN reads it) ──
        Tensor xB = new Tensor(xA.Shape, xA.DType);
        backendB.CopyFromPeer(xB, xA, backendA);
        xA.Dispose();
        Tensor tEmbB = new Tensor(tEmb.Shape, tEmb.DType);
        backendB.CopyFromPeer(tEmbB, tEmb, backendA);

        Tensor xFinalB = ForwardLayersRange(backendB, xB, tEmbB, splitBlock, _layers.Length);
        tEmbB.Dispose();

        // ── hand back to backendA: norm_out weights live there (EnumerateSharedWeights) ──
        Tensor x = new Tensor(xFinalB.Shape, xFinalB.DType);
        backendA.CopyFromPeer(x, xFinalB, backendB);
        xFinalB.Dispose();

        // ── slice off the image portion (back of the [caption, image] sequence) ──
        Tensor imageSlice;
        if (batch == 1)
        {
            imageSlice = new Tensor(new TensorShape(1, imgLen, hidden), x.DType);
            backendA.SliceRows(imageSlice, x, capLen);
        }
        else
        {
            imageSlice = SliceImageBack(x, batch, capLen, imgLen, hidden);
        }
        x.Dispose();

        Tensor finalProj = ApplyNormOut(backendA, imageSlice, tEmb, batch, imgLen);
        imageSlice.Dispose();
        tEmb.Dispose();

        Tensor velocity;
        if (batch == 1)
        {
            velocity = new Tensor(new TensorShape(1, inChannels, latentH, latentW), DType.F32);
            backendA.UnpatchifyTokens(velocity, finalProj, inChannels, hPacked, wPacked, patch, innerChannelFastest: true);
        }
        else
        {
            velocity = Unpatchify(finalProj, batch, inChannels, hPacked, wPacked, patch);
        }
        finalProj.Dispose();
        return velocity;
    }

    /// <summary>Runs main <see cref="Lumina2Block"/>s <c>[startBlock, endBlock)</c> in order — the seam a
    /// block-range shard splits on (see <see cref="ForwardSharded"/>). Unconditional dispose-and-reassign every
    /// iteration: unlike SD3's dual-stream <c>JointBlock</c>, <see cref="Lumina2Block.Forward"/> has no
    /// identity-passthrough case, so there is nothing to protect against — matches
    /// <c>Krea2Transformer.ForwardBlocksRange</c>'s shape exactly.</summary>
    private Tensor ForwardLayersRange(IBackend backend, Tensor x, Tensor tEmb, int startBlock, int endBlock)
    {
        for (int i = startBlock; i < endBlock; i++)
        {
            Tensor next = _layers[i].Forward(backend, x, tEmb, _rope);
            x.Dispose();
            x = next;
        }
        return x;
    }

    /// <summary>Computes the timestep embedding [B, adaLNEmbedDim] via sinusoidal -> Linear -> SiLU -> Linear (Lumina 2.0's TimestepEmbedding). Frequency table size matches <c>_tEmbLinear1Weight.Shape[1]</c> (the input dim of linear_1 — typically 256 for the 2304-hidden Lumina 2.0).</summary>
    private Tensor ComputeTimestepEmbedding(IBackend backend, float sigma, int batch)
    {
        int adaLNDim = _config.AdaLNEmbedDim;
        // Sinusoidal embedding dim is the input dim of linear_1, NOT the output dim. This is the
        // standard diffusers TimestepEmbedding pattern: time_proj produces [B, frequency_emb] with
        // frequency_emb=256, then linear_1 maps [256 -> adaLNDim].
        int freqEmbDim = (int)_tEmbLinear1Weight!.Shape[1];
        float scaled = sigma * _config.TimestepScale;

        TensorShape sinShape = new TensorShape(batch, freqEmbDim);
        Tensor sinEmb = new Tensor(sinShape, DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinEmb, scaled, batch, freqEmbDim);

        TensorShape midShape = new TensorShape(batch, adaLNDim);
        Tensor m1 = new Tensor(midShape, DType.F32);
        backend.Linear(m1, sinEmb, _tEmbLinear1Weight!, _tEmbLinear1Bias);
        sinEmb.Dispose();

        Tensor m1Act = new Tensor(midShape, DType.F32);
        backend.Silu(m1Act, m1);
        m1.Dispose();

        TensorShape outShape = new TensorShape(batch, adaLNDim);
        Tensor tEmb = new Tensor(outShape, DType.F32);
        backend.Linear(tEmb, m1Act, _tEmbLinear2Weight!, _tEmbLinear2Bias);
        m1Act.Dispose();

        return tEmb;
    }

    /// <summary>Caption embedding: RMSNorm(cap_embedder.0) + Linear(cap_embedder.1, with bias) -> [B, capLen, hidden].</summary>
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

    /// <summary>Patchify: [B, C, H, W] -> [B, hPacked*wPacked, pH*pW*C]. Inner ordering (slowest-to-fastest): pH, pW, C — matches the unpatchify view-permute pattern in diffusers Lumina2.</summary>
    private static Tensor Patchify(Tensor latent, int batch, int channels, int H, int W, int patch)
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

    /// <summary>Inverse of Patchify: [B, hPacked*wPacked, pH*pW*C] -> [B, C, H, W]. Inner ordering matches Patchify (pH, pW, C).</summary>
    private static Tensor Unpatchify(Tensor packed, int batch, int channels, int hPacked, int wPacked, int patch)
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

    /// <summary>Slices the image-token portion off the BACK of the [caption, image] concatenated sequence: [B, capLen+imgLen, D] -> [B, imgLen, D].</summary>
    private static Tensor SliceImageBack(Tensor combined, int batch, int capLen, int imgLen, int dim)
    {
        int totalSeq = capLen + imgLen;
        TensorShape outShape = new TensorShape(batch, imgLen, dim);
        Tensor output = new Tensor(outShape, combined.DType);

        float* srcPtr = (float*)combined.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            long bytes = (long)imgLen * dim * sizeof(float);
            Buffer.MemoryCopy(srcPtr + b * totalSeq * dim + capLen * dim, outPtr + b * imgLen * dim, bytes, bytes);
        }
        return output;
    }

    /// <summary>norm_out (LuminaLayerNormContinuous): scale = linear_1(SiLU(t_emb)); x = LayerNorm(x) * (1 + scale); x = linear_2(x). Linear_1 maps adaLNDim -> hidden; linear_2 maps hidden -> patch_dim.</summary>
    private Tensor ApplyNormOut(IBackend backend, Tensor imageTokens, Tensor tEmb, int batch, int seqLen)
    {
        int hidden = _config.HiddenSize;
        int patchOutDim = _config.InChannels * _config.PatchSize * _config.PatchSize * _config.FramePatchSize;

        TensorShape tShape = new TensorShape(batch, _config.AdaLNEmbedDim);
        Tensor activated = new Tensor(tShape, tEmb.DType);
        backend.Silu(activated, tEmb);

        TensorShape modShape = new TensorShape(batch, hidden);
        Tensor scaleParam = new Tensor(modShape, tEmb.DType);
        backend.Linear(scaleParam, activated, _normOutLinear1Weight!, _normOutLinear1Bias);
        activated.Dispose();

        // LayerNorm-no-affine on image tokens. Diffusers LuminaLayerNormContinuous defaults to
        // LayerNorm with elementwise_affine=False (eps=1e-5 typically — the same as the rest of the
        // model's norm_eps).
        TensorShape seqShape = new TensorShape(batch, seqLen, hidden);
        Tensor normed = new Tensor(seqShape, imageTokens.DType);
        Tensor modulated;
        if (batch == 1)
        {
            backend.LayerNormNoAffine(normed, imageTokens, _config.NormEps);
            Tensor scalePlus1 = new Tensor(scaleParam.Shape, DType.F32);
            backend.AddScalar(scalePlus1, scaleParam, 1.0f);
            modulated = new Tensor(seqShape, imageTokens.DType);
            backend.AffineBroadcastLastDim(modulated, normed, scalePlus1, null);
            scalePlus1.Dispose();
        }
        else
        {
            DiTUtils.LayerNormNoAffine(normed, imageTokens, batch, seqLen, hidden, _config.NormEps);
            modulated = new Tensor(seqShape, imageTokens.DType);
            float* normPtr = (float*)normed.DataPointer;
            float* scalePtr = (float*)scaleParam.DataPointer;
            float* outPtr = (float*)modulated.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                int condBase = b * hidden;
                for (int sIdx = 0; sIdx < seqLen; sIdx++)
                {
                    int seqOff = (b * seqLen + sIdx) * hidden;
                    for (int d = 0; d < hidden; d++)
                    {
                        outPtr[seqOff + d] = normPtr[seqOff + d] * (1.0f + scalePtr[condBase + d]);
                    }
                }
            }
        }
        normed.Dispose();
        scaleParam.Dispose();

        TensorShape outShape = new TensorShape(batch, seqLen, patchOutDim);
        Tensor projected = new Tensor(outShape, imageTokens.DType);
        backend.Linear(projected, modulated, _normOutLinear2Weight!, _normOutLinear2Bias);
        modulated.Dispose();

        return projected;
    }

    /// <summary>Builds caption-only position IDs for the context_refiner stack. Lumina 2.0 uses
    /// pos_start=0: every caption slot gets a unique frame index 0..capLen-1 with h=w=0.</summary>
    private static Tensor BuildCaptionPositionIds(int capLen)
    {
        TensorShape shape = new TensorShape(capLen, 3);
        Tensor posIds = new Tensor(shape, DType.F32);
        float* ptr = (float*)posIds.DataPointer;
        for (int t = 0; t < capLen; t++)
        {
            ptr[t * 3 + 0] = t;
            ptr[t * 3 + 1] = 0f;
            ptr[t * 3 + 2] = 0f;
        }
        return posIds;
    }

    /// <summary>Builds image-only position IDs (used for the noise_refiner pre-merge pass). Each
    /// image patch token gets (frame_start, row, col) in axis order [frame, h, w].</summary>
    private static Tensor BuildImagePositionIds(int hPacked, int wPacked, int imageFrameStart)
    {
        int imgLen = hPacked * wPacked;
        TensorShape shape = new TensorShape(imgLen, 3);
        Tensor posIds = new Tensor(shape, DType.F32);
        float* ptr = (float*)posIds.DataPointer;
        for (int row = 0; row < hPacked; row++)
        {
            for (int col = 0; col < wPacked; col++)
            {
                int idx = row * wPacked + col;
                ptr[idx * 3 + 0] = imageFrameStart;
                ptr[idx * 3 + 1] = row;
                ptr[idx * 3 + 2] = col;
            }
        }
        return posIds;
    }

    /// <summary>Builds joint position IDs for the [caption, image] concatenated sequence. Captions
    /// occupy slots [0, capLen) with positions (0..capLen-1, 0, 0); image patches occupy [capLen,
    /// capLen+imgLen) with positions (capLen, row, col). Mirrors diffusers
    /// <c>Lumina2RotaryPosEmbed.__call__</c>.</summary>
    private static Tensor BuildJointPositionIds(int capLen, int hPacked, int wPacked)
    {
        int imgLen = hPacked * wPacked;
        int totalLen = capLen + imgLen;
        TensorShape shape = new TensorShape(totalLen, 3);
        Tensor posIds = new Tensor(shape, DType.F32);
        float* ptr = (float*)posIds.DataPointer;

        for (int t = 0; t < capLen; t++)
        {
            ptr[t * 3 + 0] = t;
            ptr[t * 3 + 1] = 0f;
            ptr[t * 3 + 2] = 0f;
        }

        for (int row = 0; row < hPacked; row++)
        {
            for (int col = 0; col < wPacked; col++)
            {
                int idx = capLen + row * wPacked + col;
                ptr[idx * 3 + 0] = capLen;
                ptr[idx * 3 + 1] = row;
                ptr[idx * 3 + 2] = col;
            }
        }
        return posIds;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _tEmbLinear1Weight = null;
            _tEmbLinear1Bias = null;
            _tEmbLinear2Weight = null;
            _tEmbLinear2Bias = null;
            _capEmbedderNormWeight = null;
            _capEmbedderLinearWeight = null;
            _capEmbedderLinearBias = null;
            _xEmbedderWeight = null;
            _xEmbedderBias = null;
            _normOutLinear1Weight = null;
            _normOutLinear1Bias = null;
            _normOutLinear2Weight = null;
            _normOutLinear2Bias = null;
            _refinedCapA?.Dispose();
            _refinedCapA = null;
            _refinedCapB?.Dispose();
            _refinedCapB = null;
        }
        GC.SuppressFinalize(this);
    }
}
