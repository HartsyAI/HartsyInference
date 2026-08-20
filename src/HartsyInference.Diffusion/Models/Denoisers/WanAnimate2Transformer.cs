using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Per-block driving-stream self-attention K/V for one chunk, built once by
/// <see cref="WanAnimate2Transformer.EncodeDriving"/> and read by every denoise step of both CFG branches. Stored
/// <b>pre-RoPE</b>: the driving RoPE table is applied on read, so the table can never outlive the resolution it was
/// derived from. This is the port's dominant memory cost — 2 × blocks × refSeq × dim floats.</summary>
public sealed class WanAnimate2DrivingCache : IDisposable
{
    private readonly Tensor[] _k;
    private readonly Tensor[] _v;
    private int _disposed;

    internal WanAnimate2DrivingCache(Tensor[] k, Tensor[] v, int frames, int tokensPerFrame, (int T, int H, int W) genGrid)
    {
        _k = k;
        _v = v;
        Frames = frames;
        TokensPerFrame = tokensPerFrame;
        GenGrid = genGrid;
    }

    /// <summary>Driving latent frames — always exactly one fewer than the generation stream's.</summary>
    public int Frames { get; }

    /// <summary>Tokens per latent frame (<c>gridH · gridW</c>).</summary>
    public int TokensPerFrame { get; }

    /// <summary>The generation token grid this cache was built against. <see cref="WanAnimate2Transformer.Forward"/>
    /// refuses a mismatch rather than silently splicing a differently-shaped driving stream.</summary>
    public (int T, int H, int W) GenGrid { get; }

    /// <summary>Blocks covered — must equal the DiT's layer count.</summary>
    public int BlockCount => _k.Length;

    internal Tensor K(int block) => _k[block];

    internal Tensor V(int block) => _v[block];

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in _k) t.Dispose();
        foreach (Tensor t in _v) t.Dispose();
    }
}

/// <summary>Wan-Animate-2 DiT: the Wan2.1 I2V-14B backbone driven by a video instead of a motion extractor. The
/// driving video's latents run through the SAME 40 blocks once per chunk at a fixed timestep of 1
/// (<see cref="EncodeDriving"/>), and their per-block self-attention K/V are spliced frame-aligned into the
/// generation stream: generated latent frame <c>j</c> attends every generated token plus ONLY driving frame
/// <c>j-1</c>, with frame 0 (the reference-image slot) getting none. Everything else — patch embed, umT5 path,
/// <c>img_emb</c>, blocks, head, unpatchify — is Wan i2v verbatim, so <see cref="WanVideoBlock"/> is reused as-is.
///
/// <para>Three reference behaviours ComfyUI's restatement omits are implemented here, which makes ComfyUI an invalid
/// parity target for any of them: block index 9 is skipped entirely on the unconditional pass (so ComfyUI diverges
/// at any cfg &gt; 1), <see cref="WanVideoConfig.Animate2LogScale"/> biases the score band of generation latent
/// frame 1, and the driving RoPE table's horizontal offset is derived per call from the current grid instead of
/// being resolved once over a <c>-1</c> sentinel and cached on the module.</para></summary>
public sealed unsafe class WanAnimate2Transformer : IStreamableDenoiser, IDisposable
{
    /// <summary>The block the reference skips on the unconditional pass. Fires only at cfg &gt; 1, so it never runs
    /// for the distillation build.</summary>
    public const int UncondSkipBlockIndex = 9;

    /// <summary>Timestep the driving stream is anchored to, independent of the denoise loop's.</summary>
    public const float DrivingTimestep = 1.0f;

    private readonly WanVideoConfig _config;
    private readonly WanVideoBlock[] _blocks;
    private readonly WanImageEmbedder _imgEmbedder;
    private readonly WanRope _rope;
    private readonly WanForwardCaches _caches = new();
    // A cache of its own for the driving table: one shared cache holds both tables, and its cap-eviction disposes
    // EVERY entry — so a second fetch inside one forward could free the generation table still in use.
    private readonly WanForwardCaches _refRopeCache = new();
    private readonly int _patchVec;
    private int _disposed;

    private Tensor? _logScaleBiasGen, _logScaleBiasSpliced;
    private (int Hw, int Seq) _logScaleKey = (-1, -1);

    private Tensor? _patchW2d, _patchB;
    private Tensor? _projOutW, _projOutB, _finalScaleShift;
    private Tensor? _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _timeProjW, _timeProjB;
    private Tensor? _textW1, _textB1, _textW2, _textB2;

    public WanAnimate2Transformer(WanVideoConfig config)
    {
        _config = config;
        _blocks = new WanVideoBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++) _blocks[i] = new WanVideoBlock(config, crossAttnNorm: true);
        _imgEmbedder = new WanImageEmbedder(config.Eps);
        _rope = new WanRope(config.HeadDim, config.RopeTheta, config.RopeMaxSeqLen);
        _patchVec = config.InChannels * config.PatchSize.T * config.PatchSize.H * config.PatchSize.W;
    }

    public WanVideoConfig Config => _config;

    /// <summary>Driving-stream RoPE offsets for a generation grid of width <paramref name="genGridW"/> patches:
    /// <c>t = 1</c> aligns driving frame <c>j</c> with generation frame <c>j+1</c> past the reference slot, and the
    /// horizontal shift parks the driving tokens in their own strip so they never collide positionally with
    /// generation tokens. The reference stores <c>-1</c> and resolves it over the module on the first forward, which
    /// leaves the previous resolution's offset in place after a resize; this is a function of the caller's grid.</summary>
    public static (int T, int H, int W) RefRopeOffsets(int genGridW) => (1, 0, genGridW);

    /// <summary>Loads the converted (post-<c>WanVideoCheckpointConverter</c>) diffusers-named weights. Animate-2 adds
    /// no module of its own, so this is exactly the Wan i2v weight set.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _patchW2d = WanDitOps.Reshape2d(w["patch_embedding.weight"], _config.InnerDim, _patchVec);
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
        foreach (Tensor t in EnumerateSharedWeights()) yield return t;
        for (int i = 0; i < _blocks.Length; i++) foreach (Tensor t in _blocks[i].EnumerateWeights()) yield return t;
    }

    /// <inheritdoc/>
    public IEnumerable<Tensor> EnumerateSharedWeights()
    {
        foreach (Tensor? t in new[] { _patchW2d, _patchB, _projOutW, _projOutB, _finalScaleShift,
            _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _timeProjW, _timeProjB, _textW1, _textB1, _textW2, _textB2 })
            if (t is not null) yield return t;
        foreach (Tensor t in _imgEmbedder.EnumerateWeights()) yield return t;
    }

    /// <inheritdoc/>
    public int BlockCount => _blocks.Length;

    /// <inheritdoc/>
    public IStreamingBlock GetBlock(int idx) => _blocks[idx];

    /// <inheritdoc/>
    public Action<int>? BeforeBlockForward { get; set; }

    /// <summary>Runs the driving video's latents through all blocks once, at <see cref="DrivingTimestep"/>, caching
    /// each block's pre-RoPE self-attention K/V. Rebuilt per chunk (and never per CFG branch — the driving stream is
    /// outside the guidance loop, so it never sees a negative prompt).</summary>
    /// <param name="drivingLatent"><c>[1, z, Tref, H, W]</c> VAE latents of the driving video. The 36 input channels
    /// are assembled here as <c>[latent | ones(4) | latent]</c> — the same latent twice, with an all-ones mask
    /// because every driving frame is known.</param>
    /// <param name="genGrid">The generation stream's token grid, which fixes the driving RoPE offsets.</param>
    public WanAnimate2DrivingCache EncodeDriving(IBackend backend, Tensor drivingLatent, Tensor encoder,
        Tensor? clipImageEmbeds, (int T, int H, int W) genGrid)
    {
        ArgumentNullException.ThrowIfNull(drivingLatent);
        (int pt, int ph, int pw) = _config.PatchSize;
        int frames = (int)drivingLatent.Shape[2] / pt;
        int gh = (int)drivingLatent.Shape[3] / ph, gw = (int)drivingLatent.Shape[4] / pw;
        if (frames != genGrid.T - 1)
            throw new ArgumentException($"Driving stream has {frames} latent frames; expected {genGrid.T - 1} (generation frames minus the reference slot).", nameof(drivingLatent));
        if (gh != genGrid.H || gw != genGrid.W)
            throw new ArgumentException($"Driving grid {gh}×{gw} does not match the generation grid {genGrid.H}×{genGrid.W}.", nameof(drivingLatent));

        int dim = _config.InnerDim;
        int seq = frames * gh * gw;
        using Tensor conditioned = BuildDrivingChannels(drivingLatent);
        Tensor hidden = WanDitOps.Patchify(backend, conditioned, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchB);

        (Tensor cos, Tensor sin) = RefRopeCosSin(frames, gh, gw, genGrid.W);
        (Tensor temb, Tensor timestepProj) = WanDitOps.ConditionTimeGroups(backend, [DrivingTimestep], _config.FreqDim, dim,
            _timeEmb1W!, _timeEmb1B, _timeEmb2W!, _timeEmb2B, _timeProjW!, _timeProjB);
        temb.Dispose();   // the driving stream has no head; only the per-block modulation is used
        (Tensor context, int imageContextLen, bool ownsContext) = BuildContext(backend, encoder, clipImageEmbeds, dim);

        Tensor[] keys = new Tensor[_blocks.Length];
        Tensor[] values = new Tensor[_blocks.Length];
        Tensor cur = hidden;
        try
        {
            for (int i = 0; i < _blocks.Length; i++)
            {
                BeforeBlockForward?.Invoke(i);
                WanAnimate2KvCapture capture = new WanAnimate2KvCapture();
                Tensor next = _blocks[i].Forward(backend, cur, context, timestepProj, _rope, cos, sin, seq,
                    imageContextLen: imageContextLen, selfAttnKvCapture: capture);
                cur.Dispose();
                cur = next;
                keys[i] = capture.K!;
                values[i] = capture.V!;
            }
        }
        catch
        {
            foreach (Tensor? t in keys) t?.Dispose();
            foreach (Tensor? t in values) t?.Dispose();
            throw;
        }
        finally
        {
            cur.Dispose();
            timestepProj.Dispose();
            if (ownsContext) context.Dispose();
        }
        return new WanAnimate2DrivingCache(keys, values, frames, gh * gw, genGrid);
    }

    /// <summary>Velocity prediction for one CFG branch. <paramref name="latent"/> is the assembled
    /// <c>[1, 36, T, H, W]</c> generation input (noise ⊕ i2v mask ⊕ conditioning latent); latent frame 0 is the
    /// reference-image slot, which is denoised and then discarded by the caller.</summary>
    /// <param name="unconditional">The negative branch, which skips block <see cref="UncondSkipBlockIndex"/>.</param>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor encoder, float timestep,
        WanAnimate2DrivingCache driving, Tensor? clipImageEmbeds = null, bool unconditional = false)
    {
        ArgumentNullException.ThrowIfNull(driving);
        (int pt, int ph, int pw) = _config.PatchSize;
        int gt = (int)latent.Shape[2] / pt, gh = (int)latent.Shape[3] / ph, gw = (int)latent.Shape[4] / pw;
        int hw = gh * gw, s = gt * hw, dim = _config.InnerDim;
        if (driving.GenGrid != (gt, gh, gw))
            throw new ArgumentException($"Driving cache was built for grid {driving.GenGrid}, not {(gt, gh, gw)}.", nameof(driving));
        if (driving.BlockCount != _blocks.Length)
            throw new ArgumentException($"Driving cache covers {driving.BlockCount} blocks; this DiT has {_blocks.Length}.", nameof(driving));

        Tensor hidden = WanDitOps.Patchify(backend, latent, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchB);
        (Tensor cos, Tensor sin) = _caches.RopeCosSin((gt, gh, gw, 0), () => _rope.BuildCosSin(gt, gh, gw));
        (Tensor refCos, Tensor refSin) = RefRopeCosSin(driving.Frames, gh, gw, gw);
        (Tensor temb, Tensor timestepProj) = WanDitOps.ConditionTimeGroups(backend, [timestep], _config.FreqDim, dim,
            _timeEmb1W!, _timeEmb1B, _timeEmb2W!, _timeEmb2B, _timeProjW!, _timeProjB);
        (Tensor context, int imageContextLen, bool ownsContext) = BuildContext(backend, encoder, clipImageEmbeds, dim);

        (Tensor? biasGen, Tensor? biasSpliced) = LogScaleBias(hw, s);
        using WanAnimate2SelfAttnContext ctx = new WanAnimate2SelfAttnContext
        {
            RefCos = refCos,
            RefSin = refSin,
            GenFrames = gt,
            TokensPerFrame = hw,
            LogScaleBiasGen = biasGen,
            LogScaleBiasSpliced = biasSpliced,
            KeyBuffer = new Tensor(new TensorShape(1, _config.NumHeads, s + hw, _config.HeadDim), DType.F32),
            ValueBuffer = new Tensor(new TensorShape(1, _config.NumHeads, s + hw, _config.HeadDim), DType.F32),
        };

        Tensor cur = hidden;
        for (int i = 0; i < _blocks.Length; i++)
        {
            if (unconditional && i == UncondSkipBlockIndex) continue;
            BeforeBlockForward?.Invoke(i);
            ctx.DrivingK = driving.K(i);
            ctx.DrivingV = driving.V(i);
            Tensor next = _blocks[i].Forward(backend, cur, context, timestepProj, _rope, cos, sin, s,
                imageContextLen: imageContextLen, animate2: ctx);
            cur.Dispose();
            cur = next;
        }
        timestepProj.Dispose();
        if (ownsContext) context.Dispose();

        Tensor projected = WanDitOps.FinalLayer(backend, cur, temb, _finalScaleShift!, _projOutW!, _projOutB, s, dim, _config.Eps, s);
        cur.Dispose();
        temb.Dispose();
        Tensor outVel = WanDitOps.Unpatchify(projected, _config.OutChannels, gt, gh, gw, _config.PatchSize);
        projected.Dispose();
        return outVel;
    }

    /// <summary>The driving table, memoized under a key that includes the derived horizontal offset — a resolution
    /// change is a different key, so no stale offset can survive one.</summary>
    private (Tensor Cos, Tensor Sin) RefRopeCosSin(int frames, int gh, int gw, int genGridW)
    {
        (int offT, int offH, int offW) = RefRopeOffsets(genGridW);
        return _refRopeCache.RopeCosSin((frames, gh, gw, offW),
            () => _rope.BuildCosSin(frames, gh, gw, offT, offH, offW));
    }

    /// <summary>Additive score bias over the <c>[hw, 2hw)</c> key band — the keys of generation latent frame 1 — for
    /// every query, matching the reference's <c>score_mod</c> (which has no query-side condition). Null for the base
    /// build's <c>log_scale = 0</c>, which then takes the unmasked attention path.</summary>
    private (Tensor? Gen, Tensor? Spliced) LogScaleBias(int hw, int s)
    {
        if (_config.Animate2LogScale == 0f) return (null, null);
        if (_logScaleKey == (hw, s)) return (_logScaleBiasGen, _logScaleBiasSpliced);
        _logScaleBiasGen?.Dispose();
        _logScaleBiasSpliced?.Dispose();
        _logScaleBiasGen = BuildLogScaleBias(hw, s, _config.Animate2LogScale);
        _logScaleBiasSpliced = BuildLogScaleBias(hw, s + hw, _config.Animate2LogScale);
        _logScaleKey = (hw, s);
        return (_logScaleBiasGen, _logScaleBiasSpliced);
    }

    /// <summary>The <c>[hw, 2hw)</c> band of <paramref name="keys"/> gets <paramref name="logScale"/>, every other
    /// key 0. <paramref name="queries"/> is one latent frame's token count, which is also the band stride.</summary>
    internal static Tensor BuildLogScaleBias(int queries, int keys, float logScale)
    {
        Tensor bias = new Tensor(new TensorShape(queries, keys), DType.F32);
        float* p = (float*)bias.DataPointer;
        new Span<float>(p, queries * keys).Clear();
        int bandEnd = Math.Min(2 * queries, keys);
        for (int q = 0; q < queries; q++)
            for (int k = queries; k < bandEnd; k++) p[(long)q * keys + k] = logScale;
        return bias;
    }

    /// <summary>Assembles the driving stream's 36 channels: <c>[latent(16) | ones(4) | latent(16)]</c>.</summary>
    private Tensor BuildDrivingChannels(Tensor drivingLatent)
    {
        int z = (int)drivingLatent.Shape[1], t = (int)drivingLatent.Shape[2];
        int h = (int)drivingLatent.Shape[3], w = (int)drivingLatent.Shape[4];
        const int maskChannels = 4;
        if (2 * z + maskChannels != _config.InChannels)
            throw new ArgumentException($"Driving latent has {z} channels; {_config.InChannels} in-channels needs {(_config.InChannels - maskChannels) / 2}.", nameof(drivingLatent));
        Tensor source = drivingLatent.DType == DType.F32 ? drivingLatent : drivingLatent.CastTo(DType.F32);
        Tensor outT = new Tensor(new TensorShape([1L, _config.InChannels, t, h, w]), DType.F32);
        long plane = (long)t * h * w;
        long latentBytes = z * plane * 4;
        float* src = (float*)source.DataPointer, dst = (float*)outT.DataPointer;
        Buffer.MemoryCopy(src, dst, latentBytes, latentBytes);
        new Span<float>(dst + z * plane, (int)(maskChannels * plane)).Fill(1f);
        Buffer.MemoryCopy(src, dst + (z + maskChannels) * plane, latentBytes, latentBytes);
        if (!ReferenceEquals(source, drivingLatent)) source.Dispose();
        return outT;
    }

    /// <summary>The cross-attention context for one stream: its own umT5 projection, with its own CLIP embedding's
    /// 257 image tokens prepended. Both streams share this one weight set.</summary>
    private (Tensor Context, int ImageContextLen, bool Owned) BuildContext(IBackend backend, Tensor encoder, Tensor? clipImageEmbeds, int dim)
    {
        Tensor textProj = _caches.TextProj(backend, encoder, dim, _textW1!, _textB1, _textW2!, _textB2);
        if (clipImageEmbeds is null || !_imgEmbedder.IsLoaded) return (textProj, 0, false);
        Tensor imgProj = _imgEmbedder.Forward(backend, clipImageEmbeds, dim);
        int imageContextLen = (int)imgProj.Shape[0];
        Tensor context = WanDitOps.ConcatRows(imgProj, textProj, dim);
        imgProj.Dispose();
        return (context, imageContextLen, true);
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key) { Tensor t = w[key]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _logScaleBiasGen?.Dispose(); _logScaleBiasSpliced?.Dispose();
            _logScaleBiasGen = _logScaleBiasSpliced = null;
            _logScaleKey = (-1, -1);
            _patchW2d = _patchB = _projOutW = _projOutB = _finalScaleShift = null;
            _timeEmb1W = _timeEmb1B = _timeEmb2W = _timeEmb2B = _timeProjW = _timeProjB = null;
            _textW1 = _textB1 = _textW2 = _textB2 = null;
            _imgEmbedder.Clear();
        }
        GC.SuppressFinalize(this);
    }
}
