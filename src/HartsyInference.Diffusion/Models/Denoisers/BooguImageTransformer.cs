using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Boogu-Image diffusion transformer (<c>BooguImageTransformer2DModel</c>). An OmniGen2/Lumina-Image-2.0
/// lineage joint transformer with a dual-stream stage and a reference-image edit path. See
/// <c>docs/Research/BOOGU_IMAGE.md</c> for the full architecture.
///
/// <para>Reuses <see cref="OmniGen2Block"/> for the single-stream, noise-refiner, ref-image-refiner (modulated) and
/// context-refiner (non-modulated) blocks — they are byte-for-byte the same Lumina sandwich-norm block — and
/// <see cref="BooguImageDoubleBlock"/> for the new dual-stream blocks. RoPE tables are precomputed once per forward
/// (via <see cref="OmniGen2Rope.BuildTableFromPositions"/>) and shared across every block, which also lets the edit
/// path express its reference-image <c>pe_shift</c> position offsets.</para>
///
/// <para>Forward (per guidance condition, batch = 1): context-refine the caption → patch-embed + refine the noise
/// image (and reference image for edits) → 8 dual-stream blocks → fuse <c>[instruct, ref, noise]</c> → 32 single-stream
/// blocks → <c>LuminaLayerNormContinuous</c> output norm on the noise-image tail → unpatchify to <c>[B, 16, H, W]</c>.</para></summary>
public sealed unsafe class BooguImageTransformer : IDisposable
{
    private readonly BooguImageConfig _config;
    private readonly OmniGen2Rope _rope;

    private readonly BooguImageSingleBlock[] _contextRefiner;
    private readonly BooguImageSingleBlock[] _noiseRefiner;
    private readonly BooguImageSingleBlock[] _refImageRefiner;
    private readonly BooguImageDoubleBlock[] _doubleBlocks;
    private readonly BooguImageSingleBlock[] _singleBlocks;

    private Tensor? _xEmbedderW, _xEmbedderB;
    private Tensor? _refPatchEmbedderW, _refPatchEmbedderB;
    private Tensor? _timeEmbed1W, _timeEmbed1B, _timeEmbed2W, _timeEmbed2B;
    private Tensor? _captionNormW, _captionLinearW, _captionLinearB;
    private Tensor? _normOutLin1W, _normOutLin1B, _normOutLin2W, _normOutLin2B;
    private Tensor? _imageIndexEmbedding;

    private int _disposed;

    // ── Step-invariant per-generation caches. ──
    // The RoPE tables are pure functions of the sequence layout (capLen, grid, refs) — previously the four
    // host float[] builds ran EVERY forward and each block re-applied them via a host loop that D2H-drained
    // Q AND K. Cached here as device tensors in the WanRopeInterleaved layout. The context-refined caption is
    // timestep-independent (context refiner takes no temb) — cached on the instructionHidden reference (the
    // loader's prompt cache reuses refs, keeping this warm across gens); a >8-entry sweep guards prompt churn.
    private (int capLen, int hPacked, int wPacked, string refSig) _tableKey = (-1, -1, -1, "");
    private Tensor? _jointCosT, _jointSinT, _ctxCosT, _ctxSinT, _combCosT, _combSinT, _noiseCosT, _noiseSinT;
    private Tensor? _refCosT, _refSinT;
    private readonly Dictionary<Tensor, Tensor> _captionCache = new(ReferenceEqualityComparer.Instance);

    /// <summary>Creates a Boogu-Image transformer with the geometry in <paramref name="config"/>
    /// (see <see cref="BooguImageConfig.V01"/>).</summary>
    public BooguImageTransformer(BooguImageConfig config)
    {
        _config = config;
        if (config.HeadDim != config.AxesDimRope[0] + config.AxesDimRope[1] + config.AxesDimRope[2])
            throw new ArgumentException("sum(axes_dim_rope) must equal head_dim.");
        _rope = new OmniGen2Rope(config.AxesDimRope, config.RopeTheta);

        int ffn = config.FfnInnerDim;
        int cond = config.ConditioningDim;

        _contextRefiner = new BooguImageSingleBlock[config.NumRefinerLayers];
        _noiseRefiner = new BooguImageSingleBlock[config.NumRefinerLayers];
        _refImageRefiner = new BooguImageSingleBlock[config.NumRefinerLayers];
        for (int i = 0; i < config.NumRefinerLayers; i++)
        {
            _contextRefiner[i] = NewBlock(config, ffn, cond, modulation: false);
            _noiseRefiner[i] = NewBlock(config, ffn, cond, modulation: true);
            _refImageRefiner[i] = NewBlock(config, ffn, cond, modulation: true);
        }

        _doubleBlocks = new BooguImageDoubleBlock[config.NumDoubleStreamLayers];
        for (int i = 0; i < config.NumDoubleStreamLayers; i++)
            _doubleBlocks[i] = new BooguImageDoubleBlock(config.HiddenSize, config.NumAttentionHeads,
                config.NumKvHeads, config.HeadDim, ffn, cond, config.NormEps, config.QkNormEps);

        _singleBlocks = new BooguImageSingleBlock[config.NumSingleStreamLayers];
        for (int i = 0; i < config.NumSingleStreamLayers; i++)
            _singleBlocks[i] = NewBlock(config, ffn, cond, modulation: true);
    }

    private static BooguImageSingleBlock NewBlock(BooguImageConfig c, int ffn, int cond, bool modulation) =>
        new BooguImageSingleBlock(c.HiddenSize, c.NumAttentionHeads, c.NumKvHeads, c.HeadDim, ffn, cond, modulation,
            c.NormEps, c.QkNormEps);

    public BooguImageConfig Config => _config;

    /// <summary>Loads weights from a diffusers-style key dict (prefix already stripped to bare keys).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _xEmbedderW = w["x_embedder.weight"];
        w.TryGetValue("x_embedder.bias", out _xEmbedderB);
        w.TryGetValue("ref_image_patch_embedder.weight", out _refPatchEmbedderW);
        w.TryGetValue("ref_image_patch_embedder.bias", out _refPatchEmbedderB);

        _timeEmbed1W = w["time_caption_embed.timestep_embedder.linear_1.weight"];
        _timeEmbed1B = w["time_caption_embed.timestep_embedder.linear_1.bias"];
        _timeEmbed2W = w["time_caption_embed.timestep_embedder.linear_2.weight"];
        _timeEmbed2B = w["time_caption_embed.timestep_embedder.linear_2.bias"];

        _captionNormW = TensorCasts.EnsureF32(w["time_caption_embed.caption_embedder.0.weight"]);
        _captionLinearW = w["time_caption_embed.caption_embedder.1.weight"];
        _captionLinearB = w["time_caption_embed.caption_embedder.1.bias"];

        _normOutLin1W = w["norm_out.linear_1.weight"];
        _normOutLin1B = w["norm_out.linear_1.bias"];
        _normOutLin2W = w["norm_out.linear_2.weight"];
        _normOutLin2B = w["norm_out.linear_2.bias"];

        w.TryGetValue("image_index_embedding", out _imageIndexEmbedding);

        for (int i = 0; i < _contextRefiner.Length; i++) _contextRefiner[i].LoadWeights(w, $"context_refiner.{i}");
        for (int i = 0; i < _noiseRefiner.Length; i++) _noiseRefiner[i].LoadWeights(w, $"noise_refiner.{i}");
        for (int i = 0; i < _refImageRefiner.Length; i++) _refImageRefiner[i].LoadWeights(w, $"ref_image_refiner.{i}");
        for (int i = 0; i < _doubleBlocks.Length; i++) _doubleBlocks[i].LoadWeights(w, $"double_stream_layers.{i}");
        for (int i = 0; i < _singleBlocks.Length; i++) _singleBlocks[i].LoadWeights(w, $"single_stream_layers.{i}");
    }

    /// <summary>Enumerates every weight tensor for GPU preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top =
        [
            _xEmbedderW, _xEmbedderB, _refPatchEmbedderW, _refPatchEmbedderB,
            _timeEmbed1W, _timeEmbed1B, _timeEmbed2W, _timeEmbed2B,
            _captionNormW, _captionLinearW, _captionLinearB,
            _normOutLin1W, _normOutLin1B, _normOutLin2W, _normOutLin2B, _imageIndexEmbedding,
        ];
        foreach (Tensor? t in top) if (t is not null) yield return t;
        foreach (BooguImageSingleBlock b in _contextRefiner) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (BooguImageSingleBlock b in _noiseRefiner) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (BooguImageSingleBlock b in _refImageRefiner) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (BooguImageDoubleBlock b in _doubleBlocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (BooguImageSingleBlock b in _singleBlocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Predicts the flow-match velocity for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="latent">Noise latent <c>[1, in_channels, H, W]</c>.</param>
    /// <param name="timestep">Flow-match timestep <c>t'</c> in <c>[0, 1]</c>.</param>
    /// <param name="instructionHidden">Qwen3-VL final hidden states <c>[1, T, instruction_feat_dim]</c>.</param>
    /// <param name="refLatents">Optional reference-image latents <c>[1, in_channels, Hr, Wr]</c> for editing (null for T2I).</param>
    /// <returns>Predicted velocity <c>[1, in_channels, H, W]</c>.</returns>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor instructionHidden,
        IReadOnlyList<Tensor>? refLatents = null)
    {
        ThrowIfDisposed();
        if (latent.Shape.Rank != 4 || latent.Shape[0] != 1)
            throw new ArgumentException($"Latent must be [1, C, H, W], got {latent.Shape}.", nameof(latent));

        int inChannels = (int)latent.Shape[1];
        int latentH = (int)latent.Shape[2];
        int latentW = (int)latent.Shape[3];
        int patch = _config.PatchSize;
        if (latentH % patch != 0 || latentW % patch != 0)
            throw new ArgumentException($"Latent {latentH}x{latentW} not divisible by patch {patch}.", nameof(latent));
        int hPacked = latentH / patch;
        int wPacked = latentW / patch;
        int outChannels = _config.OutChannels ?? inChannels;

        Tensor imgFlat = DiTUtils.PatchifyNCHW(latent, patch);
        Tensor packedVel = ForwardCore(backend, imgFlat, timestep, instructionHidden, refLatents, hPacked, wPacked);
        imgFlat.Dispose();

        // Device unpatchify — Boogu's (py·p+px)·C + c token inner order IS innerChannelFastest; the old
        // DiTUtils.UnpatchifyToNCHW host loop forced a full D2H drain of the projection every forward.
        Tensor velocity = new Tensor(new TensorShape(1, outChannels, latentH, latentW), DType.F32);
        backend.UnpatchifyTokens(velocity, packedVel, outChannels, hPacked, wPacked, patch, innerChannelFastest: true);
        packedVel.Dispose();
        return velocity;
    }

    /// <summary>Packed-IO T2I forward: <paramref name="packedLatent"/> is <c>[1, hP·wP, p²·in_channels]</c>
    /// (the <see cref="DiTUtils.PatchifyNCHW"/> token layout) and the return is the packed velocity
    /// <c>[1, hP·wP, p²·out_channels]</c>. Lets the pipeline keep the latent in token space across the whole
    /// denoise loop (patchify once, unpatchify once) — the NCHW <see cref="Forward"/> re-ran the host
    /// PatchifyNCHW (a full D2H drain of the latent) on every forward. Same math, no refs (T2I only).</summary>
    public Tensor ForwardPacked(IBackend backend, Tensor packedLatent, float timestep, Tensor instructionHidden,
        int hPacked, int wPacked)
    {
        ThrowIfDisposed();
        return ForwardCore(backend, packedLatent, timestep, instructionHidden, null, hPacked, wPacked);
    }

    /// <summary>Shared forward body over the packed token latent <c>[1, imgLen, p²·in_channels]</c>
    /// (not consumed — caller owns it). Returns the packed velocity <c>[1, imgLen, p²·out_channels]</c>.</summary>
    private Tensor ForwardCore(IBackend backend, Tensor imgFlatIn, float timestep, Tensor instructionHidden,
        IReadOnlyList<Tensor>? refLatents, int hPacked, int wPacked)
    {
        if (instructionHidden.Shape.Rank != 3 || instructionHidden.Shape[0] != 1)
            throw new ArgumentException($"instructionHidden must be [1, T, dim], got {instructionHidden.Shape}.", nameof(instructionHidden));

        int batch = 1;
        int patch = _config.PatchSize;
        int hidden = _config.HiddenSize;
        int cond = _config.ConditioningDim;
        int capLen = (int)instructionHidden.Shape[1];
        int imgLen = hPacked * wPacked;
        if (imgFlatIn.Shape.Rank != 3 || imgFlatIn.Shape[1] != imgLen)
            throw new ArgumentException($"Packed latent must be [1, {imgLen}, p²·C], got {imgFlatIn.Shape}.", nameof(imgFlatIn));

        // ── timestep embedding ──
        Tensor temb = ComputeTimestepEmbedding(backend, timestep, batch, cond);

        // ── reference image grids (edit) ──
        int refCount = refLatents?.Count ?? 0;
        int[] refHTok = new int[refCount];
        int[] refWTok = new int[refCount];
        int totalRefLen = 0;
        for (int j = 0; j < refCount; j++)
        {
            Tensor r = refLatents![j];
            refHTok[j] = (int)r.Shape[2] / patch;
            refWTok[j] = (int)r.Shape[3] / patch;
            totalRefLen += refHTok[j] * refWTok[j];
        }
        int combinedImgLen = totalRefLen + imgLen;
        int jointLen = capLen + combinedImgLen;

        // ── RoPE tables (caption | refs | noise, with pe_shift offsets) — step-invariant per layout, so the
        //    host trig builds + device expansion run once per (capLen, grid, refs) instead of every forward ──
        string refSig = refCount == 0 ? "" : string.Join(',', refHTok) + "x" + string.Join(',', refWTok);
        (int, int, int, string) tableKey = (capLen, hPacked, wPacked, refSig);
        if (_tableKey != tableKey)
        {
            DisposeRopeTables();
            (int[] jt, int[] jh, int[] jw) = BuildJointPositionIds(capLen, refHTok, refWTok, hPacked, wPacked);
            (float[] jointCos, float[] jointSin) = _rope.BuildTableFromPositions(jt, jh, jw);
            (float[] ctxCos, float[] ctxSin) = _rope.BuildTableFromPositions(jt[..capLen], jh[..capLen], jw[..capLen]);
            (float[] combCos, float[] combSin) = _rope.BuildTableFromPositions(jt[capLen..], jh[capLen..], jw[capLen..]);
            (float[] noiseCos, float[] noiseSin) = _rope.BuildTableFromPositions(
                jt[(capLen + totalRefLen)..], jh[(capLen + totalRefLen)..], jw[(capLen + totalRefLen)..]);
            (_jointCosT, _jointSinT) = _rope.ExpandToDeviceTables(jointCos, jointSin);
            (_ctxCosT, _ctxSinT) = _rope.ExpandToDeviceTables(ctxCos, ctxSin);
            (_combCosT, _combSinT) = _rope.ExpandToDeviceTables(combCos, combSin);
            (_noiseCosT, _noiseSinT) = _rope.ExpandToDeviceTables(noiseCos, noiseSin);
            if (refCount > 0)
            {
                (float[] refCos, float[] refSin) = _rope.BuildTableFromPositions(
                    jt[capLen..(capLen + totalRefLen)], jh[capLen..(capLen + totalRefLen)], jw[capLen..(capLen + totalRefLen)]);
                (_refCosT, _refSinT) = _rope.ExpandToDeviceTables(refCos, refSin);
            }
            _tableKey = tableKey;
        }

        // ── caption embed + context refiner (text) — timestep-independent, cached per instructionHidden ref ──
        Tensor caption;
        if (!_captionCache.TryGetValue(instructionHidden, out Tensor? cachedCaption))
        {
            caption = new Tensor(new TensorShape(batch, capLen, hidden), DType.F32);
            {
                Tensor capNorm = new Tensor(new TensorShape(batch, capLen, _config.InstructionFeatDim), DType.F32);
                backend.RmsNorm(capNorm, instructionHidden, _captionNormW!, _config.NormEps);
                backend.Linear(caption, capNorm, _captionLinearW!, _captionLinearB);
                capNorm.Dispose();
            }
            for (int i = 0; i < _contextRefiner.Length; i++)
            {
                Tensor next = _contextRefiner[i].Forward(backend, caption, _ctxCosT!, _ctxSinT!, null);
                caption.Dispose();
                caption = next;
            }
            if (_captionCache.Count > 8) SweepCaptionCache();
            // Host-materialize so a later FreeActivations can't revert the cache to stale host memory.
            backend.OffloadActivation(caption);
            _captionCache[instructionHidden] = caption;
            cachedCaption = caption;
        }
        // The double-block loop consumes (disposes) its instruct input — hand it a per-forward working copy.
        {
            caption = new Tensor(new TensorShape(batch, capLen, hidden), DType.F32);
            backend.CopyInto(caption, cachedCaption);
        }

        // ── noise image patch embed + refine (input already in packed token layout; caller owns it) ──
        Tensor noiseImg = new Tensor(new TensorShape(batch, imgLen, hidden), DType.F32);
        backend.Linear(noiseImg, imgFlatIn, _xEmbedderW!, _xEmbedderB);
        for (int i = 0; i < _noiseRefiner.Length; i++)
        {
            Tensor next = _noiseRefiner[i].Forward(backend, noiseImg, _noiseCosT!, _noiseSinT!, temb);
            noiseImg.Dispose();
            noiseImg = next;
        }

        // ── reference image patch embed + refine (edit) → combined image [ref, noise] ──
        Tensor combinedImg;
        if (refCount > 0)
        {
            Tensor refTokens = BuildRefImageTokens(backend, refLatents!, refHTok, refWTok, totalRefLen, hidden, patch);
            for (int i = 0; i < _refImageRefiner.Length; i++)
            {
                Tensor next = _refImageRefiner[i].Forward(backend, refTokens, _refCosT!, _refSinT!, temb);
                refTokens.Dispose();
                refTokens = next;
            }
            combinedImg = DiTUtils.ConcatAlongSeqDim(refTokens, noiseImg);
            refTokens.Dispose();
            noiseImg.Dispose();
        }
        else
        {
            combinedImg = noiseImg;
        }

        // ── dual-stream stage ──
        Tensor instruct = caption;
        Tensor image = combinedImg;
        for (int i = 0; i < _doubleBlocks.Length; i++)
        {
            (Tensor newImage, Tensor newInstruct) = _doubleBlocks[i].Forward(backend, image, instruct,
                _jointCosT!, _jointSinT!, _combCosT!, _combSinT!, capLen, temb);
            image.Dispose();
            instruct.Dispose();
            image = newImage;
            instruct = newInstruct;
        }

        // ── fuse [instruct, image] → single-stream stack ──
        Tensor joint = DiTUtils.ConcatAlongSeqDim(instruct, image);
        instruct.Dispose();
        image.Dispose();
        for (int i = 0; i < _singleBlocks.Length; i++)
        {
            Tensor next = _singleBlocks[i].Forward(backend, joint, _jointCosT!, _jointSinT!, temb);
            joint.Dispose();
            joint = next;
        }

        // ── keep the noise-image tail, output norm, unpatchify ──
        // GPU-resident row slice of the noise image tokens (the trailing imgLen rows of the joint sequence).
        Tensor noiseTail = new Tensor(new TensorShape(batch, imgLen, hidden), DType.F32);
        backend.SliceRows(noiseTail, joint, capLen + totalRefLen);
        joint.Dispose();

        Tensor normed = ApplyOutputNorm(backend, noiseTail, temb, batch, imgLen, hidden, cond);
        noiseTail.Dispose();
        temb.Dispose();

        int outChannels = _config.OutChannels ?? _config.InChannels;
        Tensor projOut = new Tensor(new TensorShape(batch, imgLen, patch * patch * outChannels), DType.F32);
        backend.Linear(projOut, normed, _normOutLin2W!, _normOutLin2B);
        normed.Dispose();

        return projOut;
    }

    /// <summary>Patch-embeds the reference latents, concatenates them into one <c>[1, totalRefLen, hidden]</c> tensor,
    /// and adds the per-image <c>image_index_embedding[j]</c> to each image's tokens.</summary>
    private Tensor BuildRefImageTokens(IBackend backend, IReadOnlyList<Tensor> refLatents, int[] refHTok, int[] refWTok,
        int totalRefLen, int hidden, int patch)
    {
        Tensor combined = new Tensor(new TensorShape(1, totalRefLen, hidden), DType.F32);
        float* dst = (float*)combined.DataPointer;
        float* idx = _imageIndexEmbedding is null ? null : (float*)_imageIndexEmbedding.DataPointer;
        int shift = 0;
        for (int j = 0; j < refLatents.Count; j++)
        {
            int len = refHTok[j] * refWTok[j];
            Tensor flat = DiTUtils.PatchifyNCHW(refLatents[j], patch);
            Tensor emb = new Tensor(new TensorShape(1, len, hidden), DType.F32);
            backend.Linear(emb, flat, _refPatchEmbedderW!, _refPatchEmbedderB);
            flat.Dispose();
            float* src = (float*)emb.DataPointer;
            for (int s = 0; s < len; s++)
                for (int d = 0; d < hidden; d++)
                {
                    float add = idx is null ? 0f : idx[j * hidden + d];
                    dst[(shift + s) * hidden + d] = src[s * hidden + d] + add;
                }
            emb.Dispose();
            shift += len;
        }
        return combined;
    }

    /// <summary>Builds the joint-sequence position ids <c>(time, height, width)</c> for <c>[caption | refs | noise]</c>,
    /// matching <c>rope.py</c>: caption token <c>i → (i,i,i)</c>; each reference image's tokens get a running
    /// time-axis offset <c>pe_shift</c> (starts at <c>capLen</c>, advances by <c>max(refH,refW)</c>); the noise image
    /// uses the offset after all references.</summary>
    private static (int[] t, int[] h, int[] w) BuildJointPositionIds(int capLen, int[] refHTok, int[] refWTok,
        int hPacked, int wPacked)
    {
        int totalRef = 0;
        for (int j = 0; j < refHTok.Length; j++) totalRef += refHTok[j] * refWTok[j];
        int total = capLen + totalRef + hPacked * wPacked;
        int[] t = new int[total];
        int[] h = new int[total];
        int[] w = new int[total];

        for (int i = 0; i < capLen; i++) { t[i] = i; h[i] = i; w[i] = i; }

        int peShift = capLen;
        int pos = capLen;
        for (int j = 0; j < refHTok.Length; j++)
        {
            for (int r = 0; r < refHTok[j]; r++)
                for (int c = 0; c < refWTok[j]; c++)
                {
                    t[pos] = peShift; h[pos] = r; w[pos] = c; pos++;
                }
            peShift += Math.Max(refHTok[j], refWTok[j]);
        }

        for (int r = 0; r < hPacked; r++)
            for (int c = 0; c < wPacked; c++)
            {
                t[pos] = peShift; h[pos] = r; w[pos] = c; pos++;
            }
        return (t, h, w);
    }

    /// <summary>Sinusoidal(t·scale) → Linear(256→cond) → SiLU → Linear(cond→cond). Returns <c>temb [B, cond]</c>.</summary>
    private Tensor ComputeTimestepEmbedding(IBackend backend, float timestep, int batch, int cond)
    {
        const int freqDim = 256;
        Tensor sin = new Tensor(new TensorShape(batch, freqDim), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sin, timestep * _config.TimestepScale, batch, freqDim);

        Tensor proj1 = new Tensor(new TensorShape(batch, cond), DType.F32);
        backend.Linear(proj1, sin, _timeEmbed1W!, _timeEmbed1B);
        sin.Dispose();
        Tensor act = new Tensor(new TensorShape(batch, cond), DType.F32);
        backend.Silu(act, proj1);
        proj1.Dispose();
        Tensor proj2 = new Tensor(new TensorShape(batch, cond), DType.F32);
        backend.Linear(proj2, act, _timeEmbed2W!, _timeEmbed2B);
        act.Dispose();
        return proj2;
    }

    /// <summary>LuminaLayerNormContinuous: <c>scale = Linear_1(SiLU(temb))</c>; <c>x = LayerNorm(x)·(1+scale)</c>.
    /// (linear_2 is applied separately by the caller as proj_out.) GPU-resident: <c>LayerNormNoAffine</c> then
    /// <c>AddScalar(scale, 1)</c> + <c>AffineBroadcastLastDim</c> (no shift) — bit-for-bit the old CPU affine.</summary>
    private Tensor ApplyOutputNorm(IBackend backend, Tensor x, Tensor temb, int batch, int seqLen, int hidden, int cond)
    {
        Tensor act = new Tensor(new TensorShape(batch, cond), DType.F32);
        backend.Silu(act, temb);
        Tensor scale = new Tensor(new TensorShape(batch, hidden), DType.F32);
        backend.Linear(scale, act, _normOutLin1W!, _normOutLin1B);
        act.Dispose();

        Tensor ln = new Tensor(new TensorShape(batch, seqLen, hidden), DType.F32);
        backend.LayerNormNoAffine(ln, x, _config.OutNormEps);

        Tensor scalePlus1 = new Tensor(new TensorShape(batch, hidden), DType.F32);
        backend.AddScalar(scalePlus1, scale, 1.0f);
        scale.Dispose();
        Tensor output = new Tensor(new TensorShape(batch, seqLen, hidden), DType.F32);
        backend.AffineBroadcastLastDim(output, ln, scalePlus1, null);
        ln.Dispose();
        scalePlus1.Dispose();
        return output;
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    /// <summary>Disposes the cached device RoPE tables (layout change / teardown).</summary>
    private void DisposeRopeTables()
    {
        _jointCosT?.Dispose(); _jointSinT?.Dispose();
        _ctxCosT?.Dispose(); _ctxSinT?.Dispose();
        _combCosT?.Dispose(); _combSinT?.Dispose();
        _noiseCosT?.Dispose(); _noiseSinT?.Dispose();
        _refCosT?.Dispose(); _refSinT?.Dispose();
        _jointCosT = _jointSinT = _ctxCosT = _ctxSinT = _combCosT = _combSinT = _noiseCosT = _noiseSinT = null;
        _refCosT = _refSinT = null;
        _tableKey = (-1, -1, -1, "");
    }

    private void SweepCaptionCache()
    {
        foreach (Tensor t in _captionCache.Values) t.Dispose();
        _captionCache.Clear();
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            DisposeRopeTables();
            SweepCaptionCache();
            _xEmbedderW = _xEmbedderB = _refPatchEmbedderW = _refPatchEmbedderB = null;
            _timeEmbed1W = _timeEmbed1B = _timeEmbed2W = _timeEmbed2B = null;
            _captionNormW = _captionLinearW = _captionLinearB = null;
            _normOutLin1W = _normOutLin1B = _normOutLin2W = _normOutLin2B = null;
            _imageIndexEmbedding = null;
        }
        GC.SuppressFinalize(this);
    }
}
