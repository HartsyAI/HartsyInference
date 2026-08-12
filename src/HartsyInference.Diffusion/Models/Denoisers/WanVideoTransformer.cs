using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Wan-Video DiT (<c>WanTransformer3DModel</c>, Wan2.2 TI2V-5B), ported from diffusers. B=1 over a VAE latent <c>[1,48,T,H,W]</c>: Conv3d <c>(1,2,2)</c> patchify → 30 blocks (self-attn + per-head 3D RoPE / cross-attn to umT5 / FFN, 6-param AdaLN, FP32 LayerNorms) → final AdaLN + <c>proj_out</c> + unpatchify → <c>[1,48,T,H,W]</c>. Reuses backend ops + <see cref="DiTUtils"/> + (pipeline) the <c>Wan22VaeDecoder</c> + T5 encoder. See <c>docs/Research/WAN_VIDEO_ARCHITECTURE.md</c>.</summary>
public sealed unsafe class WanVideoTransformer : IDisposable
{
    private readonly WanVideoConfig _config;
    private readonly WanVideoBlock[] _blocks;
    private readonly WanRope _rope;
    private readonly int _patchVec;     // in_channels · pt · ph · pw
    private int _disposed;

    private Tensor? _patchW2d, _patchB;        // [inner, patchVec], [inner]
    private static int _fwdCounter;            // diagnostic: counts ForwardCore calls (VRAM logging)
    private Tensor? _projOutW, _projOutB;      // [out·pt·ph·pw, inner]
    private Tensor? _finalScaleShift;          // [2, inner]
    private Tensor? _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B;   // time_embedder
    private Tensor? _timeProjW, _timeProjB;    // → 6·inner
    private Tensor? _textW1, _textB1, _textW2, _textB2;   // text_embedder
    // I2V image embedder (condition_embedder.image_embedder), shared with the Animate DiT.
    private readonly WanImageEmbedder _imgEmbedder;

    // Per-generation caches for the step-invariant conditioning work (recomputed 2×/step before):
    // RoPE cos/sin keyed by the latent grid, and the projected text(+image) context keyed by encoder identity.
    private (int T, int H, int W) _ropeKey = (-1, -1, -1);
    private Tensor? _cosC, _sinC;
    private readonly Dictionary<Tensor, (Tensor Ctx, int ImageContextLen, Tensor? ImgKey)> _ctxCache = new(ReferenceEqualityComparer.Instance);

    public WanVideoTransformer(WanVideoConfig config)
    {
        _config = config;
        _blocks = new WanVideoBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++) _blocks[i] = new WanVideoBlock(config, crossAttnNorm: true);
        _rope = new WanRope(config.HeadDim, config.RopeTheta, config.RopeMaxSeqLen);
        _imgEmbedder = new WanImageEmbedder(config.Eps);
        _patchVec = config.InChannels * config.PatchSize.T * config.PatchSize.H * config.PatchSize.W;
    }

    public WanVideoConfig Config => _config;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        // patch_embedding.weight [inner, in, pt, ph, pw] → [inner, patchVec] (contiguous).
        Tensor pw = w["patch_embedding.weight"];
        _patchW2d = WanDitOps.Reshape2d(pw, _config.InnerDim, _patchVec);
        w.TryGetValue("patch_embedding.bias", out _patchB);
        _projOutW = w["proj_out.weight"]; w.TryGetValue("proj_out.bias", out _projOutB);
        _finalScaleShift = LoadF32(w, "scale_shift_table");
        _timeEmb1W = w["condition_embedder.time_embedder.linear_1.weight"]; w.TryGetValue("condition_embedder.time_embedder.linear_1.bias", out _timeEmb1B);
        _timeEmb2W = w["condition_embedder.time_embedder.linear_2.weight"]; w.TryGetValue("condition_embedder.time_embedder.linear_2.bias", out _timeEmb2B);
        _timeProjW = w["condition_embedder.time_proj.weight"]; w.TryGetValue("condition_embedder.time_proj.bias", out _timeProjB);
        _textW1 = w["condition_embedder.text_embedder.linear_1.weight"]; w.TryGetValue("condition_embedder.text_embedder.linear_1.bias", out _textB1);
        _textW2 = w["condition_embedder.text_embedder.linear_2.weight"]; w.TryGetValue("condition_embedder.text_embedder.linear_2.bias", out _textB2);
        _imgEmbedder.TryLoadWeights(w);
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"blocks.{i}");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _patchW2d, _patchB, _projOutW, _projOutB, _finalScaleShift,
            _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _timeProjW, _timeProjB, _textW1, _textB1, _textW2, _textB2 })
            if (t is not null) yield return t;
        foreach (Tensor t in _imgEmbedder.EnumerateWeights()) yield return t;
        for (int i = 0; i < _blocks.Length; i++) foreach (Tensor t in _blocks[i].EnumerateWeights()) yield return t;
    }

    /// <summary>Velocity prediction. <paramref name="latent"/> is <c>[1, inChannels, T, H, W]</c>; <paramref name="encoder"/> is raw umT5 features <c>[L, textDim]</c>. Returns <c>[1, outChannels, T, H, W]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor encoder, float timestep, DeviceFeatureCache? stepCache = null,
        CpForwardContext? cp = null) =>
        Forward(backend, latent, encoder, [timestep], stepCache, cp);

    /// <summary>I2V velocity prediction with CLIP image conditioning. <paramref name="imageEmbeds"/> is the CLIP
    /// penultimate hidden state <c>[seqImg, imageDim]</c>; it is projected by the image embedder and cross-attended in
    /// every block alongside the text context.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor encoder, float[] timesteps, Tensor imageEmbeds, DeviceFeatureCache? stepCache = null) =>
        ForwardCore(backend, latent, encoder, timesteps, imageEmbeds, stepCache);

    /// <summary>Velocity prediction with per-latent-frame timesteps (the diffusers <c>expand_timesteps</c> TI2V path —
    /// I2V conditions the first latent frame at timestep 0 while the rest denoise). <paramref name="timesteps"/> is
    /// either one shared value or one value per latent frame group (<c>T / patch_t</c>); each frame's tokens get that
    /// frame's AdaLN modulation in every block and in the final layer.
    /// <para><paramref name="cp"/> switches to the context-parallel rank path: the token sequence is sliced to the
    /// rank's frame-aligned range, per-block self-attention K/V are exchanged with the peer rank via
    /// <see cref="CpKvExchange"/>, and the return value is the LOCAL projected rows <c>[Sr, outVec]</c> — the caller
    /// gathers ranks and unpatchifies. Null keeps the single-backend path byte-identical.</para></summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor encoder, float[] timesteps, DeviceFeatureCache? stepCache = null,
        CpForwardContext? cp = null) =>
        ForwardCore(backend, latent, encoder, timesteps, null, stepCache, cp);

    private Tensor ForwardCore(IBackend backend, Tensor latent, Tensor encoder, float[] timesteps, Tensor? imageEmbeds,
        DeviceFeatureCache? stepCache = null, CpForwardContext? cp = null)
    {
        int t = (int)latent.Shape[2], hh = (int)latent.Shape[3], ww = (int)latent.Shape[4];
        (int pt, int ph, int pw) = _config.PatchSize;
        int gt = t / pt, gh = hh / ph, gw = ww / pw;
        int s = gt * gh * gw;
        int dim = _config.InnerDim;
        int g = timesteps.Length;
        if (g != 1 && g != gt)
            throw new ArgumentException($"timesteps must have 1 or {gt} (latent frame groups) entries; got {g}.", nameof(timesteps));
        int tokensPerGroup = s / g;
        if (cp is not null)
        {
            if (stepCache is not null)
                throw new InvalidOperationException("Context parallelism excludes the step cache — the pipeline must gate it off.");
            if (cp.Plan.TotalTokens != s || cp.Plan.Frames != gt)
                throw new ArgumentException($"CP plan ({cp.Plan.Frames}f × {cp.Plan.TokensPerFrame}) does not match the latent grid ({gt}×{gh}×{gw}).", nameof(cp));
        }

        BuildRopeCache(gt, gh, gw);
        (Tensor cos, Tensor sin) = (_cosC!, _sinC!);

        WanVideoDebugDump.Dump("latent_in", latent);   // raw transformer input, so the Python reference recomputes every stage
        Tensor hidden = WanDitOps.Patchify(backend, latent, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchB);   // [S, dim]
        WanVideoDebugDump.Dump("patch_embed", hidden);
        WanVideoDebugDump.Dump("in_encoder", encoder);
        if (WanVideoDebugDump.Enabled)
        {
            WanVideoDebugDump.DumpValues("timesteps", timesteps);
            if (imageEmbeds is not null) WanVideoDebugDump.Dump("clip_embeds", imageEmbeds);
        }

        // Context parallel: this rank keeps only its frame-aligned token range. Hidden rows slice on-device;
        // cos/sin slice host-side (they are host-authoritative absolute-position rows, so a rank slice keeps the
        // global positions). G>1 per-frame timesteps slice to the rank's frame range — the frame-aligned split
        // means group boundaries never straddle ranks, so per-group modulation stays exact with the GLOBAL
        // tokensPerGroup.
        int sEff = s;
        Tensor? cosLocal = null, sinLocal = null;
        float[] tsEff = timesteps;
        if (cp is not null)
        {
            CpRankRange range = cp.Plan.Ranks[cp.Rank];
            sEff = range.TokenCount;
            Tensor local = new Tensor(new TensorShape(sEff, dim), DType.F32);
            backend.SliceRows(local, hidden, range.TokenStart);
            hidden.Dispose();
            hidden = local;
            cosLocal = SliceHostRows(cos, range.TokenStart, sEff);
            sinLocal = SliceHostRows(sin, range.TokenStart, sEff);
            (cos, sin) = (cosLocal, sinLocal);
            if (g != 1) tsEff = timesteps[range.FrameStart..(range.FrameStart + range.FrameCount)];
        }

        (Tensor temb, Tensor timestepProj) = WanDitOps.ConditionTimeGroups(backend, tsEff, _config.FreqDim, dim,
            _timeEmb1W!, _timeEmb1B, _timeEmb2W!, _timeEmb2B, _timeProjW!, _timeProjB);   // [G, dim], [G, 6, dim]
        WanVideoDebugDump.Dump("cond_temb", temb);
        WanVideoDebugDump.Dump("cond_timestepProj", timestepProj);
        WanVideoDebugDump.Dump("cond_cos", cos);
        WanVideoDebugDump.Dump("cond_sin", sin);
        (Tensor encoderProj, int imageContextLen) = GetOrBuildContext(backend, encoder, imageEmbeds, dim);

        Tensor cur = hidden;
        _fwdCounter++;
        string? vramLog = Environment.GetEnvironmentVariable("HARTSY_WAN_VRAM");
        Func<Tensor, Tensor, (Tensor K, Tensor V)>? kvExchange =
            cp is null ? null : (k, v) => cp.Exchange.Exchange(cp.Rank, k, v);

        // Across-step First-Block cache (single-stream FBC; QwenImageTransformer holds the dual-stream
        // reference wiring): block 0 always runs and its output is the gate indicator. Hit ⇒ blocks 1..N−1
        // are replaced by the previous full compute's residual (device Add); miss ⇒ the block-0 output
        // survives the loop as the anchor for StoreResidual. Null stepCache ⇒ byte-identical loop.
        Tensor? cacheAnchor = null;
        int startBlock = 0;
        if (stepCache is not null && _blocks.Length > 1)
        {
            if (vramLog is not null)
                System.IO.File.AppendAllText(vramLog, $"fwd#{_fwdCounter} block 0: free {backend.FreeMemoryBytes() / (1024.0 * 1024 * 1024):F3} GB\n");
            Tensor block0 = _blocks[0].Forward(backend, cur, encoderProj, timestepProj, _rope, cos, sin, tokensPerGroup,
                imageContextLen: imageContextLen, dbg: "b0");
            cur.Dispose();
            cur = block0;
            WanVideoDebugDump.Dump("blocks.0", cur);
            startBlock = 1;
            if (!stepCache.ShouldCompute(backend, cur))
            {
                Tensor reconstructed = stepCache.ApplyResidual(backend, cur);
                cur.Dispose();
                cur = reconstructed;
                startBlock = _blocks.Length;
            }
            else
            {
                cacheAnchor = cur;
            }
        }

        for (int i = startBlock; i < _blocks.Length; i++)
        {
            if (vramLog is not null)
                System.IO.File.AppendAllText(vramLog, $"fwd#{_fwdCounter} block {i}: free {backend.FreeMemoryBytes() / (1024.0 * 1024 * 1024):F3} GB\n");
            Tensor next = _blocks[i].Forward(backend, cur, encoderProj, timestepProj, _rope, cos, sin, tokensPerGroup,
                imageContextLen: imageContextLen, dbg: i == 0 ? "b0" : null, selfAttnKvExchange: kvExchange);
            if (!ReferenceEquals(cur, cacheAnchor)) cur.Dispose();
            cur = next;
            WanVideoDebugDump.Dump($"blocks.{i}", cur);
        }

        if (cacheAnchor is not null)
        {
            stepCache!.StoreResidual(backend, cacheAnchor, cur);
            cacheAnchor.Dispose();
        }
        timestepProj.Dispose();   // cos/sin/encoderProj live in the per-generation caches — not per-forward temporaries

        Tensor projected = WanDitOps.FinalLayer(backend, cur, temb, _finalScaleShift!, _projOutW!, _projOutB, sEff, dim, _config.Eps, tokensPerGroup);
        cur.Dispose();
        temb.Dispose();
        if (cp is not null)
        {
            cosLocal!.Dispose();
            sinLocal!.Dispose();
            return projected;   // local [Sr, outVec] rows — the pipeline gathers ranks + unpatchifies
        }
        Tensor outVel = WanDitOps.Unpatchify(projected, _config.OutChannels, gt, gh, gw, _config.PatchSize);
        projected.Dispose();
        WanVideoDebugDump.DumpOutput(outVel);
        return outVel;
    }

    private void BuildRopeCache(int gt, int gh, int gw)
    {
        if (_ropeKey == (gt, gh, gw)) return;
        _cosC?.Dispose(); _sinC?.Dispose();
        (_cosC, _sinC) = _rope.BuildCosSin(gt, gh, gw);
        _ropeKey = (gt, gh, gw);
    }

    /// <summary>The projected text(+image) context is timestep-independent — cache it per encoder tensor identity
    /// (with the image-embeds identity validated on hit for I2V, where the context also folds in the CLIP image and
    /// a HOST ConcatRows) so the projections + gelu + concat run once per generation instead of 2×/step. The cached
    /// tensor is host-materialized here, so it survives the pipeline's per-step FreeActivations (device copy
    /// re-uploads on demand; the host data stays authoritative).</summary>
    private (Tensor Ctx, int ImageContextLen) GetOrBuildContext(IBackend backend, Tensor encoder, Tensor? imageEmbeds, int dim)
    {
        if (_ctxCache.TryGetValue(encoder, out (Tensor Ctx, int ImageContextLen, Tensor? ImgKey) cached)
            && ReferenceEquals(cached.ImgKey, imageEmbeds))
        {
            return (cached.Ctx, cached.ImageContextLen);
        }
        Tensor textProj = WanDitOps.TextEmbed(backend, encoder, dim, _textW1!, _textB1, _textW2!, _textB2);
        WanVideoDebugDump.Dump("cond_textProj", textProj);

        // I2V: project the CLIP image embeds and prepend them to the text context; the blocks split at imageContextLen.
        int imageContextLen = 0;
        Tensor encoderProj = textProj;
        if (imageEmbeds is not null && _imgEmbedder.IsLoaded)
        {
            Tensor imgProj = _imgEmbedder.Forward(backend, imageEmbeds, dim);
            WanVideoDebugDump.Dump("cond_imgProj", imgProj);
            imageContextLen = (int)imgProj.Shape[0];
            encoderProj = WanDitOps.ConcatRows(imgProj, textProj, dim);
            imgProj.Dispose();
            textProj.Dispose();
        }
        WanVideoDebugDump.Dump("cond_encoderProj", encoderProj);
        backend.OffloadActivation(encoderProj);   // host-materialize (see cache note above)
        if (_ctxCache.Count >= 4)   // cap: prompts/images changed across gens; drop the stale contexts
        {
            foreach ((Tensor ctx, _, _) in _ctxCache.Values) ctx.Dispose();
            _ctxCache.Clear();
        }
        // Same-encoder re-entry with a different image replaces the stale entry (dispose the old context).
        if (_ctxCache.Remove(encoder, out (Tensor Ctx, int ImageContextLen, Tensor? ImgKey) stale)) stale.Ctx.Dispose();
        _ctxCache[encoder] = (encoderProj, imageContextLen, imageEmbeds);
        return (encoderProj, imageContextLen);
    }

    /// <summary>Builds the per-generation rope + text-context caches on the CALLING thread — load-bearing before
    /// context-parallel forks: two rank threads must never race the first-touch builds of the shared
    /// <c>_cosC</c>/<c>_sinC</c>/<c>_ctxCache</c> state (after this, in-forward access is read-only cache hits).</summary>
    public void PrewarmSequenceCaches(IBackend backend, int gridT, int gridH, int gridW, params Tensor[] encoders)
    {
        BuildRopeCache(gridT, gridH, gridW);
        foreach (Tensor encoder in encoders) GetOrBuildContext(backend, encoder, null, _config.InnerDim);
    }

    /// <summary>Host row slice of a host-authoritative <c>[rows, cols]</c> tensor (the rope cos/sin tables) — no
    /// backend involvement, safe from concurrent rank threads.</summary>
    private static Tensor SliceHostRows(Tensor src, int rowStart, int rowCount)
    {
        int cols = (int)src.Shape[1];
        Tensor o = new Tensor(new TensorShape(rowCount, cols), DType.F32);
        long bytes = (long)rowCount * cols * 4;
        Buffer.MemoryCopy((float*)src.DataPointer + (long)rowStart * cols, (float*)o.DataPointer, bytes, bytes);
        return o;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key) { Tensor t = w[key]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _patchW2d = _patchB = _projOutW = _projOutB = _finalScaleShift = null;
            _timeEmb1W = _timeEmb1B = _timeEmb2W = _timeEmb2B = _timeProjW = _timeProjB = null;
            _textW1 = _textB1 = _textW2 = _textB2 = null;
            _imgEmbedder.Clear();
        }
        GC.SuppressFinalize(this);
    }
}
