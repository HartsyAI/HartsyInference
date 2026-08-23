using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>OmniGen 2 (image t2i) transformer. Wraps the unified <see cref="OmniGen2Block"/> stack:
/// 2 noise refiner + 2 context refiner + N main joint blocks (concat of <c>[text_caption_tokens,
/// image_patch_tokens]</c>), with shared 3-axis RoPE.
///
/// <para>Forward chain (mirrors <c>OmniGen2Transformer2DModel.forward</c> in diffusers'
/// <c>transformer_omnigen2.py</c>):</para>
/// <list type="number">
///   <item>Patchify the latent: reshape <c>[B, C, H, W]</c> -&gt; <c>[B, S_img, p² * C]</c> with <c>p = patch_size</c>.</item>
///   <item>Patch embed: <c>Linear((p² * C), hidden)</c>.</item>
///   <item>Caption embed: <c>Linear(text_feat_dim, hidden)</c>.</item>
///   <item>Time-caption embed: sinusoidal(<c>t * timestep_scale</c>) -&gt; SiLU MLP -&gt; <c>temb [B, conditioning_dim]</c>.</item>
///   <item>Noise refiner: 2 modulated blocks on image tokens with <c>RopeApplyMode.Image</c>, <c>timeOffset = txtSeqLen</c>.</item>
///   <item>Context refiner: 2 non-modulated blocks on text tokens with <c>RopeApplyMode.Text</c>.</item>
///   <item>Concat <c>[text, image]</c> along sequence axis.</item>
///   <item>Main blocks: <c>NumLayers</c> modulated blocks with <c>RopeApplyMode.Joint</c>.</item>
///   <item>Strip the text prefix: keep only the trailing image tokens.</item>
///   <item>Final norm: LuminaRMSNormZero (modulation-style) producing <c>[B, S_img, hidden]</c>.</item>
///   <item>Project out: <c>Linear(hidden, p² * out_channels)</c>, unpatchify back to <c>[B, C_out, H, W]</c>.</item>
/// </list>
///
/// <para>Editing / multi-image-input: pass reference-image VAE latents to the
/// <see cref="Forward(IBackend, Tensor, float, Tensor, int, IReadOnlyList{Tensor})"/> overload. Each reference is
/// patchified through the dedicated <c>ref_image_patch_embedder</c>, tagged with its <c>image_index_embedding</c>
/// row, refined per-image through the modulated <c>ref_image_refiner</c> blocks (attention isolated per reference,
/// mirroring upstream's per-reference batching), and the joint stream becomes <c>[text, ref_0..N, noise]</c> with
/// the upstream <c>pe_shift</c> 3-axis positions (each reference and the noise grid get consecutive time-axis
/// offsets of <c>max(ref_h_tokens, ref_w_tokens)</c> past the caption length).</para></summary>
public sealed unsafe class OmniGen2Transformer : IDisposable
{
    private readonly OmniGen2Config _config;
    private readonly OmniGen2Rope _rope;

    private readonly OmniGen2Block[] _noiseRefiner;
    private readonly OmniGen2Block[] _contextRefiner;
    private readonly OmniGen2Block[] _refImageRefiner;
    private readonly OmniGen2Block[] _mainBlocks;

    private Tensor? _xEmbedderWeight, _xEmbedderBias;
    private Tensor? _captionNormWeight;                       // time_caption_embed.caption_embedder.0 (RMSNorm over text_feat_dim)
    private Tensor? _captionEmbedderWeight, _captionEmbedderBias; // time_caption_embed.caption_embedder.1 (Linear text_feat_dim → hidden)
    private Tensor? _timeProj1Weight, _timeProj1Bias;        // time_caption_embed.timestep_embedder.linear_1 (256 → 1024)
    private Tensor? _timeProj2Weight, _timeProj2Bias;        // time_caption_embed.timestep_embedder.linear_2 (1024 → 1024)
    private Tensor? _normOutLinearWeight, _normOutLinearBias; // norm_out.linear_1 (AdaLN: conditioning → hidden)
    private Tensor? _projOutWeight, _projOutBias;            // norm_out.linear_2 (hidden → p²·out_channels)
    private Tensor? _imageIndexEmbedding;
    private Tensor? _refEmbedderWeight, _refEmbedderBias;    // ref_image_patch_embedder (Linear p²·C → hidden)
    private Tensor?[] _refCombinedBias = [];                 // per-slot ref_image_patch_embedder.bias + image_index_embedding[j]
    private bool _hasRefStack;

    private int _disposed;

    /// <summary>Creates an OmniGen 2 transformer. The block geometry comes entirely from
    /// <paramref name="config"/>; refer to <see cref="OmniGen2Config.V1"/> for the public release.</summary>
    public OmniGen2Transformer(OmniGen2Config config)
    {
        _config = config;
        _rope = new OmniGen2Rope(config.AxesDimRope, config.RopeTheta);

        int ffnInner = ComputeFfnInnerDim(config);
        int conditioningDim = config.ConditioningDim;

        _noiseRefiner = new OmniGen2Block[config.NumRefinerLayers];
        _contextRefiner = new OmniGen2Block[config.NumRefinerLayers];
        _refImageRefiner = new OmniGen2Block[config.NumRefinerLayers];
        _mainBlocks = new OmniGen2Block[config.NumLayers];

        for (int i = 0; i < config.NumRefinerLayers; i++)
        {
            _noiseRefiner[i] = new OmniGen2Block(
                config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads, config.HeadDim,
                ffnInner, conditioningDim, modulation: true, config.NormEps, config.QkNormEps);
            _contextRefiner[i] = new OmniGen2Block(
                config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads, config.HeadDim,
                ffnInner, conditioningDim, modulation: false, config.NormEps, config.QkNormEps);
            _refImageRefiner[i] = new OmniGen2Block(
                config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads, config.HeadDim,
                ffnInner, conditioningDim, modulation: true, config.NormEps, config.QkNormEps);
        }
        for (int i = 0; i < config.NumLayers; i++)
        {
            _mainBlocks[i] = new OmniGen2Block(
                config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads, config.HeadDim,
                ffnInner, conditioningDim, modulation: true, config.NormEps, config.QkNormEps);
        }
    }

    public OmniGen2Config Config => _config;

    /// <summary>Loads weights from a diffusers-style key dict.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        weights.TryGetValue("x_embedder.weight", out _xEmbedderWeight);
        weights.TryGetValue("x_embedder.bias", out _xEmbedderBias);

        // caption_embedder is nested under time_caption_embed: Sequential(RMSNorm(text_feat_dim), Linear(text_feat_dim → hidden)).
        _captionNormWeight = CastToF32IfNeeded(weights["time_caption_embed.caption_embedder.0.weight"]);
        weights.TryGetValue("time_caption_embed.caption_embedder.1.weight", out _captionEmbedderWeight);
        weights.TryGetValue("time_caption_embed.caption_embedder.1.bias", out _captionEmbedderBias);

        // timestep_embedder: TimestepEmbedding(256 → 1024 → 1024), SiLU between the two linears.
        weights.TryGetValue("time_caption_embed.timestep_embedder.linear_1.weight", out _timeProj1Weight);
        weights.TryGetValue("time_caption_embed.timestep_embedder.linear_1.bias", out _timeProj1Bias);
        weights.TryGetValue("time_caption_embed.timestep_embedder.linear_2.weight", out _timeProj2Weight);
        weights.TryGetValue("time_caption_embed.timestep_embedder.linear_2.bias", out _timeProj2Bias);

        // norm_out = LuminaLayerNormContinuous: linear_1 is the AdaLN scale (conditioning → hidden), the norm is a
        // non-affine LayerNorm (no weight/bias, eps 1e-6), linear_2 is the output projection (hidden → p²·out_channels).
        weights.TryGetValue("norm_out.linear_1.weight", out _normOutLinearWeight);
        weights.TryGetValue("norm_out.linear_1.bias", out _normOutLinearBias);
        weights.TryGetValue("norm_out.linear_2.weight", out _projOutWeight);
        weights.TryGetValue("norm_out.linear_2.bias", out _projOutBias);

        weights.TryGetValue("image_index_embedding", out _imageIndexEmbedding);

        // Reference-image stack (edit / multi-image input). Optional so pruned t2i-only checkpoints still load;
        // Forward with refLatents throws when absent.
        weights.TryGetValue("ref_image_patch_embedder.weight", out _refEmbedderWeight);
        weights.TryGetValue("ref_image_patch_embedder.bias", out _refEmbedderBias);
        _hasRefStack = _refEmbedderWeight is not null
            && weights.ContainsKey("ref_image_refiner.0.attn.to_q.weight");
        if (_hasRefStack)
        {
            for (int i = 0; i < _refImageRefiner.Length; i++)
                _refImageRefiner[i].LoadWeights(weights, $"ref_image_refiner.{i}");
            BuildRefCombinedBiases();
        }

        for (int i = 0; i < _noiseRefiner.Length; i++)
            _noiseRefiner[i].LoadWeights(weights, $"noise_refiner.{i}");
        for (int i = 0; i < _contextRefiner.Length; i++)
            _contextRefiner[i].LoadWeights(weights, $"context_refiner.{i}");
        for (int i = 0; i < _mainBlocks.Length; i++)
            _mainBlocks[i].LoadWeights(weights, $"layers.{i}");
    }

    /// <summary>Whether the checkpoint carried the reference-image stack (<c>ref_image_patch_embedder</c> +
    /// <c>ref_image_refiner</c>) required by the edit path.</summary>
    public bool HasRefStack => _hasRefStack;

    /// <summary>Folds <c>image_index_embedding[j]</c> into the ref patch embedder's bias so the per-reference
    /// index tag costs nothing at runtime: upstream computes <c>ref_image_patch_embedder(x) + image_index_embedding[j]</c>
    /// token-wise, which equals a Linear with bias <c>(bias + image_index_embedding[j])</c>. Kept in the embedder
    /// bias' dtype so the Linear sees a homogeneous parameter set.</summary>
    private void BuildRefCombinedBiases()
    {
        if (_refEmbedderBias is null || _imageIndexEmbedding is null)
        {
            _refCombinedBias = [];
            return;
        }
        int slots = (int)_imageIndexEmbedding.Shape[0];
        int hidden = (int)_imageIndexEmbedding.Shape[1];
        Tensor biasF32 = _refEmbedderBias.DType == DType.F32 ? _refEmbedderBias : _refEmbedderBias.CastTo(DType.F32);
        Tensor idxF32 = _imageIndexEmbedding.DType == DType.F32 ? _imageIndexEmbedding : _imageIndexEmbedding.CastTo(DType.F32);
        _refCombinedBias = new Tensor?[slots];
        float* bp = (float*)biasF32.DataPointer;
        float* ip = (float*)idxF32.DataPointer;
        for (int j = 0; j < slots; j++)
        {
            Tensor combined = new(new TensorShape(hidden), DType.F32);
            float* cp = (float*)combined.DataPointer;
            for (int d = 0; d < hidden; d++)
                cp[d] = bp[d] + ip[(long)j * hidden + d];
            if (_refEmbedderBias.DType != DType.F32)
            {
                Tensor cast = combined.CastTo(_refEmbedderBias.DType);
                combined.Dispose();
                combined = cast;
            }
            _refCombinedBias[j] = combined;
        }
        if (!ReferenceEquals(biasF32, _refEmbedderBias)) biasF32.Dispose();
        if (!ReferenceEquals(idxF32, _imageIndexEmbedding)) idxF32.Dispose();
    }

    /// <summary>Enumerates every weight tensor for GPU preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_xEmbedderWeight is not null) yield return _xEmbedderWeight;
        if (_xEmbedderBias is not null) yield return _xEmbedderBias;
        if (_captionNormWeight is not null) yield return _captionNormWeight;
        if (_captionEmbedderWeight is not null) yield return _captionEmbedderWeight;
        if (_captionEmbedderBias is not null) yield return _captionEmbedderBias;
        if (_timeProj1Weight is not null) yield return _timeProj1Weight;
        if (_timeProj1Bias is not null) yield return _timeProj1Bias;
        if (_timeProj2Weight is not null) yield return _timeProj2Weight;
        if (_timeProj2Bias is not null) yield return _timeProj2Bias;
        if (_normOutLinearWeight is not null) yield return _normOutLinearWeight;
        if (_normOutLinearBias is not null) yield return _normOutLinearBias;
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
        if (_imageIndexEmbedding is not null) yield return _imageIndexEmbedding;
        if (_refEmbedderWeight is not null) yield return _refEmbedderWeight;
        if (_refEmbedderBias is not null) yield return _refEmbedderBias;
        foreach (Tensor? combined in _refCombinedBias)
            if (combined is not null) yield return combined;

        foreach (OmniGen2Block b in _noiseRefiner)
            foreach (Tensor w in b.EnumerateWeights()) yield return w;
        foreach (OmniGen2Block b in _contextRefiner)
            foreach (Tensor w in b.EnumerateWeights()) yield return w;
        if (_hasRefStack)
        {
            foreach (OmniGen2Block b in _refImageRefiner)
                foreach (Tensor w in b.EnumerateWeights()) yield return w;
        }
        foreach (OmniGen2Block b in _mainBlocks)
            foreach (Tensor w in b.EnumerateWeights()) yield return w;
    }

    /// <summary>Forward pass — predicts velocity for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="latent">Input latent <c>[B, in_channels, H, W]</c>.</param>
    /// <param name="timestep">Flow-match sigma in <c>[0, 1]</c> (1 = pure noise, 0 = clean). OmniGen2 internally
    /// embeds <c>(1 - sigma)</c> and negates its output, so this is the raw scheduler sigma, not sigma·1000.</param>
    /// <param name="textEmbeds">Caption embeddings <c>[B, T, text_feat_dim]</c>.</param>
    /// <param name="textSeqLen">Number of valid caption tokens (textEmbeds.Shape[1]).</param>
    /// <returns>Predicted velocity <c>[B, out_channels, H, W]</c>.</returns>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor textEmbeds, int textSeqLen) =>
        Forward(backend, latent, timestep, textEmbeds, textSeqLen, refLatents: null);

    /// <summary>Forward pass with optional reference-image conditioning (the edit / multi-image-input path).
    /// <paramref name="refLatents"/> holds VAE latents <c>[1, in_channels, Hr, Wr]</c>, one per reference in
    /// picture order (max <see cref="OmniGen2Config.MaxRefImages"/>); null/empty falls back to plain t2i. The
    /// joint sequence becomes <c>[text, ref_0..N, noise]</c> and only the trailing noise tokens are projected
    /// out, per <c>OmniGen2Transformer2DModel.forward</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor textEmbeds, int textSeqLen,
        IReadOnlyList<Tensor>? refLatents)
    {
        ThrowIfDisposed();
        if (latent.Shape.Rank != 4)
            throw new ArgumentException($"Latent must be 4D [B, C, H, W], got {latent.Shape}.", nameof(latent));
        if (textEmbeds.Shape.Rank != 3)
            throw new ArgumentException($"textEmbeds must be 3D [B, T, dim], got {textEmbeds.Shape}.", nameof(textEmbeds));

        return refLatents is null || refLatents.Count == 0
            ? ForwardText2Image(backend, latent, timestep, textEmbeds, textSeqLen)
            : ForwardWithRefs(backend, latent, timestep, textEmbeds, textSeqLen, refLatents);
    }

    private Tensor ForwardText2Image(IBackend backend, Tensor latent, float timestep, Tensor textEmbeds, int textSeqLen)
    {
        int inChannels = (int)latent.Shape[1];
        int latentH = (int)latent.Shape[2];
        int latentW = (int)latent.Shape[3];
        int patch = _config.PatchSize;
        int hPacked = latentH / patch;
        int wPacked = latentW / patch;
        int outChannels = _config.OutChannels ?? inChannels;

        if (latentH % patch != 0 || latentW % patch != 0)
            throw new ArgumentException($"Latent {latentH}x{latentW} not divisible by patch {patch}.", nameof(latent));

        // NCHW convenience wrapper over the packed path: patchify once, run the packed forward, unpatchify once.
        // The packed forward already realized upstream's `return -output`, so this unpatchify does NOT negate.
        Tensor packed = DiTUtils.PatchifyNCHW(latent, patch);
        Tensor velPacked = ForwardPacked(backend, packed, timestep, textEmbeds, textSeqLen, hPacked, wPacked);
        packed.Dispose();
        Tensor velocity = new(new TensorShape(1, outChannels, latentH, latentW), DType.F32);
        backend.UnpatchifyTokens(velocity, velPacked, outChannels, hPacked, wPacked, patch, innerChannelFastest: true);
        velPacked.Dispose();
        OmniGen2DebugDump.Dump("output_velocity", velocity);
        return velocity;
    }

    /// <summary>Packed-latent forward for the drain-free denoise loop. Takes the already-patchified token latent
    /// <c>[1, imgSeqLen, p²·in_channels]</c> (never consumed — caller owns it) and returns the packed velocity
    /// <c>[1, imgSeqLen, p²·out_channels]</c>, negated per upstream's <c>return -output</c>. Skips the per-forward
    /// host patchify/unpatchify D2H drains (the <c>cpu-glue-async-race</c> crash source) so the latent stays
    /// GPU-resident across the whole sampling loop — the pipeline patchifies once, runs <see cref="CfgEulerStep"/>
    /// per step, and unpatchifies once.</summary>
    public Tensor ForwardPacked(IBackend backend, Tensor packedLatent, float timestep, Tensor textEmbeds,
        int textSeqLen, int hPacked, int wPacked)
    {
        ThrowIfDisposed();
        const int batch = 1;
        int hidden = _config.HiddenSize;
        int inChannels = _config.InChannels;
        int patch = _config.PatchSize;
        int imgSeqLen = hPacked * wPacked;
        int outChannels = _config.OutChannels ?? inChannels;
        int conditioningDim = _config.ConditioningDim;
        if (packedLatent.Shape.Rank != 3 || packedLatent.Shape[1] != imgSeqLen)
            throw new ArgumentException($"Packed latent must be [1, {imgSeqLen}, p²·C], got {packedLatent.Shape}.", nameof(packedLatent));

        // ── Patch embed (on the already-packed latent) ──
        Tensor imgTokens = new(new TensorShape(batch, imgSeqLen, hidden), DType.F32);
        backend.Linear(imgTokens, packedLatent, _xEmbedderWeight!, _xEmbedderBias);

        // ── Caption embed: RMSNorm(text_feat_dim) → Linear(text_feat_dim → hidden) ──
        Tensor txtNormed = new(textEmbeds.Shape, DType.F32);
        backend.RmsNorm(txtNormed, textEmbeds, _captionNormWeight!, _config.NormEps);
        Tensor txtTokens = new(new TensorShape(batch, textSeqLen, hidden), DType.F32);
        backend.Linear(txtTokens, txtNormed, _captionEmbedderWeight!, _captionEmbedderBias);
        txtNormed.Dispose();

        // F16 hot path (HARTSY_DIT_F16): cast both streams to F16 once; the blocks run entirely in F16. Safe now
        // that ConcatAlongSeqDim/SplitAlongSeqDim are dtype-aware (they were F32-hardcoded → 2× OOB read on the
        // F16 buffers = the intermittent segfault). No-op when the flag is off.
        imgTokens = CastStreamToAct(backend, imgTokens);
        txtTokens = CastStreamToAct(backend, txtTokens);

        Tensor temb = ComputeTimestepEmbedding(backend, timestep, batch, conditioningDim);

        for (int i = 0; i < _noiseRefiner.Length; i++)
        {
            Tensor next = _noiseRefiner[i].Forward(backend, imgTokens, _rope, RopeApplyMode.Image,
                hPacked, wPacked, timeOffset: textSeqLen, temb);
            imgTokens.Dispose();
            imgTokens = next;
        }
        for (int i = 0; i < _contextRefiner.Length; i++)
        {
            Tensor next = _contextRefiner[i].Forward(backend, txtTokens, _rope, RopeApplyMode.Text,
                hPacked: 0, wPacked: 0, timeOffset: 0, temb: null);
            txtTokens.Dispose();
            txtTokens = next;
        }
        // Device concat [text, image] (dim 1) — keeps the forward fully GPU-resident (the host ConcatAlongSeqDim
        // drained both streams to the CPU every forward, and its host DataPointer reads block CUDA-graph capture).
        Tensor joint = new(new TensorShape(batch, textSeqLen + imgSeqLen, hidden), imgTokens.DType);
        backend.Concat(joint, new[] { txtTokens, imgTokens }, dim: 1);
        txtTokens.Dispose();
        imgTokens.Dispose();
        for (int i = 0; i < _mainBlocks.Length; i++)
        {
            Tensor next = _mainBlocks[i].Forward(backend, joint, _rope, RopeApplyMode.Joint,
                hPacked, wPacked, timeOffset: 0, temb);
            joint.Dispose();
            joint = next;
        }
        return FinishOutputPacked(backend, joint, temb, prefixSeqLen: textSeqLen, batch, imgSeqLen, hidden,
            conditioningDim, outChannels, patch);
    }

    /// <summary>Packed-space output tail: strips the text prefix, applies the final AdaLN-continuous norm and
    /// proj_out (in F32), and negates (upstream <c>return -output</c>) — returning the packed velocity
    /// <c>[1, imgSeqLen, p²·out_channels]</c> without the host unpatchify. Consumes <paramref name="joint"/> and
    /// <paramref name="temb"/>.</summary>
    private Tensor FinishOutputPacked(IBackend backend, Tensor joint, Tensor temb, int prefixSeqLen, int batch,
        int imgSeqLen, int hidden, int conditioningDim, int outChannels, int patch)
    {
        // Device slice of the trailing image tokens (drop the [text] prefix) — replaces the host SplitAlongSeqDim
        // drain, keeping the tail GPU-resident and CUDA-graph-capturable.
        Tensor imgFinal = new(new TensorShape(batch, imgSeqLen, hidden), joint.DType);
        backend.SliceRows(imgFinal, joint, prefixSeqLen);
        joint.Dispose();

        if (imgFinal.DType != DType.F32)
        {
            Tensor imgFinalF32 = new(imgFinal.Shape, DType.F32);
            backend.CastToF32(imgFinalF32, imgFinal);
            imgFinal.Dispose();
            imgFinal = imgFinalF32;
        }

        Tensor normedOut = ApplyFinalNorm(backend, imgFinal, temb, batch, imgSeqLen, hidden, conditioningDim);
        imgFinal.Dispose();
        temb.Dispose();

        Tensor projOut = new(new TensorShape(batch, imgSeqLen, patch * patch * outChannels), DType.F32);
        backend.Linear(projOut, normedOut, _projOutWeight!, _projOutBias);
        normedOut.Dispose();

        // Realize upstream OmniGen2's `return -output` in packed token space (device negate); the drain-free loop
        // consumes this negated velocity directly via CfgEulerStep and unpatchifies the clean latent once at the end.
        Tensor velocity = new(projOut.Shape, DType.F32);
        backend.Scale(velocity, projOut, -1.0f);
        projOut.Dispose();
        return velocity;
    }

    /// <summary>Reference-image-conditioned forward. Joint stream = <c>[text, ref_0..N, noise]</c> per
    /// <c>OmniGen2Transformer2DModel.forward</c>: refs go through <c>ref_image_patch_embedder</c> (+ the folded
    /// <c>image_index_embedding[j]</c> bias), are refined per-reference by the modulated <c>ref_image_refiner</c>
    /// (attention isolated to each reference, matching upstream's per-reference batching), and every stream is
    /// rotated with upstream's <c>pe_shift</c> position ids: text token <c>s</c> at <c>(s,s,s)</c>; reference
    /// <c>j</c>'s grid at time-axis <c>pe_shift_j</c> where <c>pe_shift</c> starts at the caption length and
    /// advances by <c>max(ref_h_tokens, ref_w_tokens)</c> per reference; the noise grid at the final shift.</summary>
    private Tensor ForwardWithRefs(IBackend backend, Tensor latent, float timestep, Tensor textEmbeds,
        int textSeqLen, IReadOnlyList<Tensor> refLatents)
    {
        if (!_hasRefStack)
            throw new InvalidOperationException(
                "OmniGen2 edit requires the ref_image_patch_embedder / ref_image_refiner weights, which this " +
                "checkpoint does not carry (t2i-only prune?). Use the full OmniGen2 transformer checkpoint.");
        int batch = (int)latent.Shape[0];
        if (batch != 1)
            throw new ArgumentException($"Reference-conditioned forward supports batch 1, got {batch}.", nameof(latent));
        int slots = _imageIndexEmbedding is not null ? (int)_imageIndexEmbedding.Shape[0] : _config.MaxRefImages;
        if (refLatents.Count > slots)
            throw new ArgumentException(
                $"OmniGen2 supports at most {slots} reference images (image_index_embedding slots), got {refLatents.Count}.",
                nameof(refLatents));

        int inChannels = (int)latent.Shape[1];
        int latentH = (int)latent.Shape[2];
        int latentW = (int)latent.Shape[3];
        int patch = _config.PatchSize;
        int hidden = _config.HiddenSize;
        int hPacked = latentH / patch;
        int wPacked = latentW / patch;
        int imgSeqLen = hPacked * wPacked;
        int outChannels = _config.OutChannels ?? inChannels;
        int conditioningDim = _config.ConditioningDim;
        int halfDim = _rope.HeadDim / 2;

        if (latentH % patch != 0 || latentW % patch != 0)
            throw new ArgumentException($"Latent {latentH}x{latentW} not divisible by patch {patch}.", nameof(latent));

        int numRefs = refLatents.Count;
        int[] refHPacked = new int[numRefs];
        int[] refWPacked = new int[numRefs];
        int[] refLen = new int[numRefs];
        int totalRefLen = 0;
        for (int j = 0; j < numRefs; j++)
        {
            Tensor r = refLatents[j];
            if (r.Shape.Rank != 4 || r.Shape[0] != 1 || r.Shape[1] != inChannels)
                throw new ArgumentException($"refLatents[{j}] must be [1, {inChannels}, Hr, Wr], got {r.Shape}.", nameof(refLatents));
            if (r.Shape[2] % patch != 0 || r.Shape[3] % patch != 0)
                throw new ArgumentException($"refLatents[{j}] {r.Shape[2]}x{r.Shape[3]} not divisible by patch {patch}.", nameof(refLatents));
            refHPacked[j] = (int)r.Shape[2] / patch;
            refWPacked[j] = (int)r.Shape[3] / patch;
            refLen[j] = refHPacked[j] * refWPacked[j];
            totalRefLen += refLen[j];
        }

        // ── 1. Patchify + embed the noise latent (x_embedder) ──
        Tensor imgFlat = DiTUtils.PatchifyNCHW(latent, patch);
        Tensor imgTokens = new(new TensorShape(batch, imgSeqLen, hidden), DType.F32);
        backend.Linear(imgTokens, imgFlat, _xEmbedderWeight!, _xEmbedderBias);
        imgFlat.Dispose();

        // ── 2. Patchify + embed each reference (ref_image_patch_embedder; bias pre-folded with
        //       image_index_embedding[j] — see BuildRefCombinedBiases) ──
        Tensor[] refTokens = new Tensor[numRefs];
        for (int j = 0; j < numRefs; j++)
        {
            Tensor refFlat = DiTUtils.PatchifyNCHW(refLatents[j], patch);
            refTokens[j] = new Tensor(new TensorShape(1, refLen[j], hidden), DType.F32);
            Tensor? bias = j < _refCombinedBias.Length ? _refCombinedBias[j] : _refEmbedderBias;
            backend.Linear(refTokens[j], refFlat, _refEmbedderWeight!, bias ?? _refEmbedderBias);
            refFlat.Dispose();
        }

        // ── 3. Caption embed: RMSNorm(text_feat_dim) → Linear(text_feat_dim → hidden) ──
        Tensor txtNormed = new(textEmbeds.Shape, DType.F32);
        backend.RmsNorm(txtNormed, textEmbeds, _captionNormWeight!, _config.NormEps);
        Tensor txtTokens = new(new TensorShape(batch, textSeqLen, hidden), DType.F32);
        backend.Linear(txtTokens, txtNormed, _captionEmbedderWeight!, _captionEmbedderBias);
        txtNormed.Dispose();

        // ── 4. Timestep embedding ──
        Tensor temb = ComputeTimestepEmbedding(backend, timestep, batch, conditioningDim);

        // ── 5. 3-axis position ids for the full [text, ref_0..N, noise] sequence (upstream pe_shift walk) ──
        int totalSeq = textSeqLen + totalRefLen + imgSeqLen;
        int[] timeIds = new int[totalSeq];
        int[] heightIds = new int[totalSeq];
        int[] widthIds = new int[totalSeq];
        for (int s = 0; s < textSeqLen; s++)
        {
            timeIds[s] = s;
            heightIds[s] = s;
            widthIds[s] = s;
        }
        int peShift = textSeqLen;
        int offset = textSeqLen;
        for (int j = 0; j < numRefs; j++)
        {
            for (int s = 0; s < refLen[j]; s++)
            {
                int row = s / refWPacked[j];
                timeIds[offset + s] = peShift;
                heightIds[offset + s] = row;
                widthIds[offset + s] = s - row * refWPacked[j];
            }
            peShift += Math.Max(refHPacked[j], refWPacked[j]);
            offset += refLen[j];
        }
        for (int s = 0; s < imgSeqLen; s++)
        {
            int row = s / wPacked;
            timeIds[offset + s] = peShift;
            heightIds[offset + s] = row;
            widthIds[offset + s] = s - row * wPacked;
        }
        (float[] ropeCos, float[] ropeSin) = _rope.BuildTableFromPositions(timeIds, heightIds, widthIds);

        // ── 6. Noise refiner on noise tokens (their joint-sequence positions, i.e. the shifted time axis) ──
        int noiseTableOffset = (textSeqLen + totalRefLen) * halfDim;
        for (int i = 0; i < _noiseRefiner.Length; i++)
        {
            Tensor next = _noiseRefiner[i].Forward(backend, imgTokens, _rope,
                ropeCos.AsSpan(noiseTableOffset, imgSeqLen * halfDim),
                ropeSin.AsSpan(noiseTableOffset, imgSeqLen * halfDim), temb);
            imgTokens.Dispose();
            imgTokens = next;
        }

        // ── 7. Ref-image refiner: each reference refined separately (attention isolated per reference) ──
        int refTableOffset = textSeqLen * halfDim;
        for (int j = 0; j < numRefs; j++)
        {
            for (int i = 0; i < _refImageRefiner.Length; i++)
            {
                Tensor next = _refImageRefiner[i].Forward(backend, refTokens[j], _rope,
                    ropeCos.AsSpan(refTableOffset, refLen[j] * halfDim),
                    ropeSin.AsSpan(refTableOffset, refLen[j] * halfDim), temb);
                refTokens[j].Dispose();
                refTokens[j] = next;
            }
            refTableOffset += refLen[j] * halfDim;
        }

        // ── 8. Context refiner on text tokens (non-modulated, positions (s,s,s)) ──
        for (int i = 0; i < _contextRefiner.Length; i++)
        {
            Tensor next = _contextRefiner[i].Forward(backend, txtTokens, _rope,
                ropeCos.AsSpan(0, textSeqLen * halfDim), ropeSin.AsSpan(0, textSeqLen * halfDim), temb: null);
            txtTokens.Dispose();
            txtTokens = next;
        }

        // ── 9. Concat [text, ref_0..N, noise] along the sequence axis ──
        Tensor jointSeq = txtTokens;
        for (int j = 0; j < numRefs; j++)
        {
            Tensor merged = DiTUtils.ConcatAlongSeqDim(jointSeq, refTokens[j]);
            jointSeq.Dispose();
            refTokens[j].Dispose();
            jointSeq = merged;
        }
        Tensor withNoise = DiTUtils.ConcatAlongSeqDim(jointSeq, imgTokens);
        jointSeq.Dispose();
        imgTokens.Dispose();
        jointSeq = withNoise;

        // ── 10. Main joint blocks over the full sequence ──
        for (int i = 0; i < _mainBlocks.Length; i++)
        {
            Tensor next = _mainBlocks[i].Forward(backend, jointSeq, _rope, ropeCos, ropeSin, temb);
            jointSeq.Dispose();
            jointSeq = next;
        }

        // ── 11. Strip the [text, refs] prefix, final norm, proj_out, unpatchify ──
        return FinishOutput(backend, jointSeq, temb, prefixSeqLen: textSeqLen + totalRefLen, batch, imgSeqLen,
            hidden, conditioningDim, outChannels, hPacked, wPacked, patch);
    }

    /// <summary>Shared output tail: drops the first <paramref name="prefixSeqLen"/> tokens (text and, on the edit
    /// path, reference tokens), applies the LuminaLayerNormContinuous final norm, projects to
    /// <c>p²·out_channels</c>, and unpatchifies — negating per upstream's <c>return -output</c> (OmniGen2 flips the
    /// flow direction with <c>timestep = 1 − sigma</c>, so the negated velocity matches the flow-match Euler step
    /// <c>x_next = x + v·(σ_next − σ)</c>). Consumes (disposes) <paramref name="joint"/> and <paramref name="temb"/>.</summary>
    private Tensor FinishOutput(IBackend backend, Tensor joint, Tensor temb, int prefixSeqLen, int batch,
        int imgSeqLen, int hidden, int conditioningDim, int outChannels, int hPacked, int wPacked, int patch)
    {
        (Tensor prefix, Tensor imgFinal) = DiTUtils.SplitAlongSeqDim(joint, prefixSeqLen);
        prefix.Dispose();
        joint.Dispose();

        // Back to F32 for the final AdaLN norm + proj_out + Euler step (velocity precision matters across steps).
        // No-op on the F32 edit path (ForwardWithRefs never casts its stream to F16).
        if (imgFinal.DType != DType.F32)
        {
            Tensor imgFinalF32 = new(imgFinal.Shape, DType.F32);
            backend.CastToF32(imgFinalF32, imgFinal);
            imgFinal.Dispose();
            imgFinal = imgFinalF32;
        }

        Tensor normedOut = ApplyFinalNorm(backend, imgFinal, temb, batch, imgSeqLen, hidden, conditioningDim);
        imgFinal.Dispose();
        temb.Dispose();

        TensorShape projOutShape = new(batch, imgSeqLen, patch * patch * outChannels);
        Tensor projOut = new(projOutShape, DType.F32);
        backend.Linear(projOut, normedOut, _projOutWeight!, _projOutBias);
        normedOut.Dispose();

        // negate realizes upstream OmniGen2's `return -output`.
        Tensor velocity = DiTUtils.UnpatchifyToNCHW(projOut, outChannels, hPacked, wPacked, patch, negate: true);
        projOut.Dispose();

        OmniGen2DebugDump.Dump("output_velocity", velocity);
        return velocity;
    }

    /// <summary>Builds the conditioning vector <c>temb [B, conditioning_dim]</c>: sinusoidal embedding of the
    /// scaled timestep, fed through the two-layer SiLU MLP <c>time_proj.{0,2}</c>.</summary>
    private Tensor ComputeTimestepEmbedding(IBackend backend, float sigma, int batch, int conditioningDim)
    {
        // OmniGen2 (per ComfyUI/upstream forward) flips the flow-match sigma: the embedded timestep is
        // (1 - sigma), then time_proj's Timesteps applies scale=timestep_scale to a 256-dim sinusoid
        // (frequency_embedding_size=256, flip_sin_to_cos=True, downscale_freq_shift=0).
        const int FrequencyEmbeddingSize = 256;
        float scaledT = (1.0f - sigma) * _config.TimestepScale;

        TensorShape sinShape = new(batch, FrequencyEmbeddingSize);
        Tensor sinusoidal = new(sinShape, DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinusoidal, scaledT, batch, FrequencyEmbeddingSize);

        // timestep_embedder.linear_1: Linear(256 → conditioning_dim)
        TensorShape projShape = new(batch, conditioningDim);
        Tensor proj1 = new(projShape, DType.F32);
        backend.Linear(proj1, sinusoidal, _timeProj1Weight!, _timeProj1Bias);
        sinusoidal.Dispose();

        Tensor act = new(projShape, DType.F32);
        backend.Silu(act, proj1);
        proj1.Dispose();

        // timestep_embedder.linear_2: Linear(conditioning_dim → conditioning_dim)
        Tensor proj2 = new(projShape, DType.F32);
        backend.Linear(proj2, act, _timeProj2Weight!, _timeProj2Bias);
        act.Dispose();
        return proj2;
    }

    /// <summary>LuminaLayerNormContinuous final layer (mirrors diffusers <c>norm_out</c>): the AdaLN scale is
    /// <c>Linear(silu(temb))</c> via <c>norm_out.linear_1</c> (conditioning → hidden); the norm is a non-affine
    /// LayerNorm (no weight/bias, eps 1e-6); output is <c>LayerNorm(x) * (1 + scale)</c>. GPU-resident.</summary>
    private Tensor ApplyFinalNorm(IBackend backend, Tensor x, Tensor temb, int batch, int seqLen, int hidden, int conditioningDim)
    {
        TensorShape actShape = new(batch, conditioningDim);
        Tensor activated = new(actShape, DType.F32);
        backend.Silu(activated, temb);

        TensorShape scaleShape = new(batch, hidden);
        Tensor scale = new(scaleShape, DType.F32);
        backend.Linear(scale, activated, _normOutLinearWeight!, _normOutLinearBias);
        activated.Dispose();

        TensorShape rmsShape = new(batch, seqLen, hidden);
        Tensor normed = new(rmsShape, DType.F32);
        backend.LayerNormNoAffine(normed, x, FinalNormEps);

        Tensor scalePlus1 = new(scaleShape, DType.F32);
        backend.AddScalar(scalePlus1, scale, 1.0f);
        scale.Dispose();

        Tensor output = new(rmsShape, DType.F32);
        backend.AffineBroadcastLastDim(output, normed, scalePlus1, null);
        normed.Dispose();
        scalePlus1.Dispose();
        return output;
    }

    /// <summary>Epsilon for the non-affine final LayerNorm (diffusers <c>LuminaLayerNormContinuous(eps=1e-6)</c>).</summary>
    private const float FinalNormEps = 1e-6f;

    private static Tensor? CastToF32IfNeeded(Tensor? t) =>
        t is null ? null : t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    /// <summary>Casts an F32 block-input stream to the DiT activation dtype (F16 on the <c>HARTSY_DIT_F16</c> hot
    /// path, else a no-op passthrough). Disposes the source when it casts. Device-resident.</summary>
    private static Tensor CastStreamToAct(IBackend backend, Tensor f32Stream)
    {
        DType act = DiTBlocks.DitDtype.Act;
        if (act == DType.F32)
            return f32Stream;
        Tensor casted = new(f32Stream.Shape, act);
        backend.CastToF16(casted, f32Stream);
        f32Stream.Dispose();
        return casted;
    }

    private static int ComputeFfnInnerDim(OmniGen2Config config)
    {
        // Upstream LuminaFeedForward (OmniGen2TransformerBlock) is constructed with inner_dim = 4 * dim, then an
        // optional ffn_dim_multiplier, then rounded UP to multiple_of. It does NOT apply the Llama SwiGLU 2/3
        // reduction (which would give 8/3 * dim) — for V1 this yields round_up(4*2520, 256) = 10240, matching the
        // shipped feed_forward.linear_1 weight [10240, 2520]. Using 8/3*dim (=6912) under-sizes every SwiGLU
        // buffer while backend.Linear still writes N=10240 from the weight, overflowing the buffers and corrupting
        // the tail image tokens (the blocky bottom-third artifact).
        int target = 4 * config.HiddenSize;
        if (config.FfnDimMultiplier is float m) target = (int)(target * m);
        int rem = target % config.MultipleOf;
        return rem == 0 ? target : target + (config.MultipleOf - rem);
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _rope.Dispose();
            _xEmbedderWeight = _xEmbedderBias = null;
            _captionNormWeight = null;
            _captionEmbedderWeight = _captionEmbedderBias = null;
            _timeProj1Weight = _timeProj1Bias = null;
            _timeProj2Weight = _timeProj2Bias = null;
            _normOutLinearWeight = _normOutLinearBias = null;
            _projOutWeight = _projOutBias = null;
            _imageIndexEmbedding = null;
            _refEmbedderWeight = _refEmbedderBias = null;
            foreach (Tensor? combined in _refCombinedBias)
                combined?.Dispose();
            _refCombinedBias = [];
        }
        GC.SuppressFinalize(this);
    }
}
