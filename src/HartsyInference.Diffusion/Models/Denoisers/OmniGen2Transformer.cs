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
/// <para>Editing / multi-image-input paths are intentionally out of scope (t2i only). The
/// <c>image_index_embedding</c> table is loaded for checkpoint-key compatibility but not consumed.</para></summary>
public sealed unsafe class OmniGen2Transformer : IDisposable
{
    private readonly OmniGen2Config _config;
    private readonly OmniGen2Rope _rope;

    private readonly OmniGen2Block[] _noiseRefiner;
    private readonly OmniGen2Block[] _contextRefiner;
    private readonly OmniGen2Block[] _mainBlocks;

    private Tensor? _xEmbedderWeight, _xEmbedderBias;
    private Tensor? _captionNormWeight;                       // time_caption_embed.caption_embedder.0 (RMSNorm over text_feat_dim)
    private Tensor? _captionEmbedderWeight, _captionEmbedderBias; // time_caption_embed.caption_embedder.1 (Linear text_feat_dim → hidden)
    private Tensor? _timeProj1Weight, _timeProj1Bias;        // time_caption_embed.timestep_embedder.linear_1 (256 → 1024)
    private Tensor? _timeProj2Weight, _timeProj2Bias;        // time_caption_embed.timestep_embedder.linear_2 (1024 → 1024)
    private Tensor? _normOutLinearWeight, _normOutLinearBias; // norm_out.linear_1 (AdaLN: conditioning → hidden)
    private Tensor? _projOutWeight, _projOutBias;            // norm_out.linear_2 (hidden → p²·out_channels)
    private Tensor? _imageIndexEmbedding;

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
        _mainBlocks = new OmniGen2Block[config.NumLayers];

        for (int i = 0; i < config.NumRefinerLayers; i++)
        {
            _noiseRefiner[i] = new OmniGen2Block(
                config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads, config.HeadDim,
                ffnInner, conditioningDim, modulation: true, config.NormEps, config.QkNormEps);
            _contextRefiner[i] = new OmniGen2Block(
                config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads, config.HeadDim,
                ffnInner, conditioningDim, modulation: false, config.NormEps, config.QkNormEps);
        }
        for (int i = 0; i < config.NumLayers; i++)
        {
            _mainBlocks[i] = new OmniGen2Block(
                config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads, config.HeadDim,
                ffnInner, conditioningDim, modulation: true, config.NormEps, config.QkNormEps);
        }
    }

    /// <summary>Convenience accessor for the config.</summary>
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

        for (int i = 0; i < _noiseRefiner.Length; i++)
            _noiseRefiner[i].LoadWeights(weights, $"noise_refiner.{i}");
        for (int i = 0; i < _contextRefiner.Length; i++)
            _contextRefiner[i].LoadWeights(weights, $"context_refiner.{i}");
        for (int i = 0; i < _mainBlocks.Length; i++)
            _mainBlocks[i].LoadWeights(weights, $"layers.{i}");
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

        foreach (OmniGen2Block b in _noiseRefiner)
            foreach (Tensor w in b.EnumerateWeights()) yield return w;
        foreach (OmniGen2Block b in _contextRefiner)
            foreach (Tensor w in b.EnumerateWeights()) yield return w;
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
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor textEmbeds, int textSeqLen)
    {
        ThrowIfDisposed();
        if (latent.Shape.Rank != 4)
            throw new ArgumentException($"Latent must be 4D [B, C, H, W], got {latent.Shape}.", nameof(latent));
        if (textEmbeds.Shape.Rank != 3)
            throw new ArgumentException($"textEmbeds must be 3D [B, T, dim], got {textEmbeds.Shape}.", nameof(textEmbeds));

        int batch = (int)latent.Shape[0];
        int inChannels = (int)latent.Shape[1];
        int latentH = (int)latent.Shape[2];
        int latentW = (int)latent.Shape[3];
        int patch = _config.PatchSize;
        int hidden = _config.HiddenSize;
        int hPacked = latentH / patch;
        int wPacked = latentW / patch;
        int imgSeqLen = hPacked * wPacked;
        int outChannels = _config.OutChannels ?? inChannels;
        int patchVolume = patch * patch * inChannels;
        int conditioningDim = _config.ConditioningDim;

        if (latentH % patch != 0 || latentW % patch != 0)
            throw new ArgumentException($"Latent {latentH}x{latentW} not divisible by patch {patch}.", nameof(latent));

        // ── 1. Patchify latent [B, C, H, W] → [B, S_img, p²·C] ──
        Tensor imgFlat = PatchifyLatent(latent, batch, inChannels, latentH, latentW, patch);

        // ── 2. Patch embed: Linear(p²·C → hidden) ──
        TensorShape imgEmbShape = new(batch, imgSeqLen, hidden);
        Tensor imgTokens = new(imgEmbShape, DType.F32);
        backend.Linear(imgTokens, imgFlat, _xEmbedderWeight!, _xEmbedderBias);
        imgFlat.Dispose();

        // ── 3. Caption embed: RMSNorm(text_feat_dim) → Linear(text_feat_dim → hidden) ──
        Tensor txtNormed = new(textEmbeds.Shape, DType.F32);
        backend.RmsNorm(txtNormed, textEmbeds, _captionNormWeight!, _config.NormEps);
        TensorShape txtEmbShape = new(batch, textSeqLen, hidden);
        Tensor txtTokens = new(txtEmbShape, DType.F32);
        backend.Linear(txtTokens, txtNormed, _captionEmbedderWeight!, _captionEmbedderBias);
        txtNormed.Dispose();

        // ── 4. Timestep embedding: sinusoidal(t * scale) → SiLU MLP → conditioning ──
        Tensor temb = ComputeTimestepEmbedding(backend, timestep, batch, conditioningDim);

        // ── 5. Noise refiner: image tokens with image RoPE ──
        for (int i = 0; i < _noiseRefiner.Length; i++)
        {
            Tensor next = _noiseRefiner[i].Forward(backend, imgTokens, _rope, RopeApplyMode.Image,
                hPacked, wPacked, timeOffset: textSeqLen, temb);
            imgTokens.Dispose();
            imgTokens = next;
        }

        // ── 6. Context refiner: text tokens with text RoPE (no modulation) ──
        for (int i = 0; i < _contextRefiner.Length; i++)
        {
            Tensor next = _contextRefiner[i].Forward(backend, txtTokens, _rope, RopeApplyMode.Text,
                hPacked: 0, wPacked: 0, timeOffset: 0, temb: null);
            txtTokens.Dispose();
            txtTokens = next;
        }

        // ── 7. Concat [text, image] along sequence axis ──
        Tensor joint = DiTUtils.ConcatAlongSeqDim(txtTokens, imgTokens);
        txtTokens.Dispose();
        imgTokens.Dispose();

        // ── 8. Main joint blocks ──
        for (int i = 0; i < _mainBlocks.Length; i++)
        {
            Tensor next = _mainBlocks[i].Forward(backend, joint, _rope, RopeApplyMode.Joint,
                hPacked, wPacked, timeOffset: 0, temb);
            joint.Dispose();
            joint = next;
        }

        // ── 9. Strip text prefix, keep image tail ──
        (Tensor _, Tensor imgFinal) = DiTUtils.SplitAlongSeqDim(joint, textSeqLen);
        joint.Dispose();

        // ── 10. Final norm: LuminaRMSNormZero (Linear(silu(temb)) → scale, then RMSNorm * (1+scale)) ──
        Tensor normedOut = ApplyFinalNorm(backend, imgFinal, temb, batch, imgSeqLen, hidden, conditioningDim);
        imgFinal.Dispose();
        temb.Dispose();

        // ── 11. proj_out: Linear(hidden → p²·out_channels) ──
        TensorShape projOutShape = new(batch, imgSeqLen, patch * patch * outChannels);
        Tensor projOut = new(projOutShape, DType.F32);
        backend.Linear(projOut, normedOut, _projOutWeight!, _projOutBias);
        normedOut.Dispose();

        // ── 12. Unpatchify [B, S_img, p²·C_out] → [B, C_out, H, W], negating per upstream's `return -output`.
        //        OmniGen2's forward flips the flow direction (timestep = 1 - sigma) and negates the velocity so the
        //        sign matches the flow-match Euler step x_next = x + v·(σ_next − σ). ──
        Tensor velocity = UnpatchifyTokens(projOut, batch, outChannels, hPacked, wPacked, patch);
        projOut.Dispose();

        OmniGen2DebugDump.Dump("output_velocity", velocity);
        return velocity;
    }

    /// <summary>Reshapes <c>[B, C, H, W]</c> to <c>[B, S_img = (H/p)*(W/p), p²·C]</c> with channel-outer ordering
    /// inside each patch. Per upstream OmniGen2's <c>x = einops.rearrange(x, 'B C (H p) (W q) -&gt; B (H W) (p q C)')</c>:
    /// for each spatial patch, the <c>p²·C</c> features are laid out as <c>[(py, px, c) for py in range(p) for px in
    /// range(p) for c in range(C)]</c>.</summary>
    private static Tensor PatchifyLatent(Tensor latent, int batch, int channels, int height, int width, int patch)
    {
        int hPacked = height / patch;
        int wPacked = width / patch;
        int imgSeqLen = hPacked * wPacked;
        int patchVolume = patch * patch * channels;

        TensorShape outShape = new(batch, imgSeqLen, patchVolume);
        Tensor result = new(outShape, DType.F32);

        float* src = (float*)latent.DataPointer;
        float* dst = (float*)result.DataPointer;
        long chwStride = (long)channels * height * width;
        long hwStride = (long)height * width;

        for (int b = 0; b < batch; b++)
        {
            float* batchSrc = src + b * chwStride;
            float* batchDst = dst + (long)b * imgSeqLen * patchVolume;
            for (int hp = 0; hp < hPacked; hp++)
            {
                for (int wp = 0; wp < wPacked; wp++)
                {
                    long tokenIdx = (long)hp * wPacked + wp;
                    float* tokenDst = batchDst + tokenIdx * patchVolume;
                    int outIdx = 0;
                    for (int py = 0; py < patch; py++)
                    {
                        int srcRow = hp * patch + py;
                        for (int px = 0; px < patch; px++)
                        {
                            int srcCol = wp * patch + px;
                            for (int c = 0; c < channels; c++)
                            {
                                tokenDst[outIdx++] = batchSrc[c * hwStride + srcRow * width + srcCol];
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    /// <summary>Inverse of <see cref="PatchifyLatent"/>: <c>[B, S_img, p²·C_out]</c> → <c>[B, C_out, H, W]</c>.</summary>
    private static Tensor UnpatchifyTokens(Tensor tokens, int batch, int channels, int hPacked, int wPacked, int patch)
    {
        int height = hPacked * patch;
        int width = wPacked * patch;
        int imgSeqLen = hPacked * wPacked;
        int patchVolume = patch * patch * channels;

        TensorShape outShape = new(batch, channels, height, width);
        Tensor result = new(outShape, DType.F32);

        float* src = (float*)tokens.DataPointer;
        float* dst = (float*)result.DataPointer;
        long chwStride = (long)channels * height * width;
        long hwStride = (long)height * width;

        for (int b = 0; b < batch; b++)
        {
            float* batchSrc = src + (long)b * imgSeqLen * patchVolume;
            float* batchDst = dst + b * chwStride;
            for (int hp = 0; hp < hPacked; hp++)
            {
                for (int wp = 0; wp < wPacked; wp++)
                {
                    long tokenIdx = (long)hp * wPacked + wp;
                    float* tokenSrc = batchSrc + tokenIdx * patchVolume;
                    int srcIdx = 0;
                    for (int py = 0; py < patch; py++)
                    {
                        int dstRow = hp * patch + py;
                        for (int px = 0; px < patch; px++)
                        {
                            int dstCol = wp * patch + px;
                            for (int c = 0; c < channels; c++)
                            {
                                // Negate here to realize upstream OmniGen2's `return -output`.
                                batchDst[c * hwStride + dstRow * width + dstCol] = -tokenSrc[srcIdx++];
                            }
                        }
                    }
                }
            }
        }
        return result;
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

    private static int ComputeFfnInnerDim(OmniGen2Config config)
    {
        int target = (int)(8.0 / 3.0 * config.HiddenSize);
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
            _xEmbedderWeight = _xEmbedderBias = null;
            _captionNormWeight = null;
            _captionEmbedderWeight = _captionEmbedderBias = null;
            _timeProj1Weight = _timeProj1Bias = null;
            _timeProj2Weight = _timeProj2Bias = null;
            _normOutLinearWeight = _normOutLinearBias = null;
            _projOutWeight = _projOutBias = null;
            _imageIndexEmbedding = null;
        }
        GC.SuppressFinalize(this);
    }
}
