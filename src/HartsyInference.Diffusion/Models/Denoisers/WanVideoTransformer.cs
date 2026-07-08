using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

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
    public Tensor Forward(IBackend backend, Tensor latent, Tensor encoder, float timestep) =>
        Forward(backend, latent, encoder, [timestep]);

    /// <summary>I2V velocity prediction with CLIP image conditioning. <paramref name="imageEmbeds"/> is the CLIP
    /// penultimate hidden state <c>[seqImg, imageDim]</c>; it is projected by the image embedder and cross-attended in
    /// every block alongside the text context.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor encoder, float[] timesteps, Tensor imageEmbeds) =>
        ForwardCore(backend, latent, encoder, timesteps, imageEmbeds);

    /// <summary>Velocity prediction with per-latent-frame timesteps (the diffusers <c>expand_timesteps</c> TI2V path —
    /// I2V conditions the first latent frame at timestep 0 while the rest denoise). <paramref name="timesteps"/> is
    /// either one shared value or one value per latent frame group (<c>T / patch_t</c>); each frame's tokens get that
    /// frame's AdaLN modulation in every block and in the final layer.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor encoder, float[] timesteps) =>
        ForwardCore(backend, latent, encoder, timesteps, null);

    private Tensor ForwardCore(IBackend backend, Tensor latent, Tensor encoder, float[] timesteps, Tensor? imageEmbeds)
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

        if (_ropeKey != (gt, gh, gw))
        {
            _cosC?.Dispose(); _sinC?.Dispose();
            (_cosC, _sinC) = _rope.BuildCosSin(gt, gh, gw);
            _ropeKey = (gt, gh, gw);
        }
        (Tensor cos, Tensor sin) = (_cosC!, _sinC!);

        WanVideoDebugDump.Dump("latent_in", latent);   // raw transformer input, so the Python reference recomputes every stage
        Tensor hidden = WanDitOps.Patchify(backend, latent, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchB);   // [S, dim]
        WanVideoDebugDump.Dump("patch_embed", hidden);
        WanVideoDebugDump.Dump("in_encoder", encoder);

        (Tensor temb, Tensor timestepProj) = WanDitOps.ConditionTimeGroups(backend, timesteps, _config.FreqDim, dim,
            _timeEmb1W!, _timeEmb1B, _timeEmb2W!, _timeEmb2B, _timeProjW!, _timeProjB);   // [G, dim], [G, 6, dim]
        WanVideoDebugDump.Dump("cond_temb", temb);
        WanVideoDebugDump.Dump("cond_timestepProj", timestepProj);
        WanVideoDebugDump.Dump("cond_cos", cos);
        WanVideoDebugDump.Dump("cond_sin", sin);
        // The projected text(+image) context is timestep-independent — cache it per encoder tensor identity (with
        // the image-embeds identity validated on hit for I2V, where the context also folds in the CLIP image and a
        // HOST ConcatRows) so the projections + gelu + concat run once per generation instead of 2×/step. The
        // cached tensor is host-materialized below, so it survives the pipeline's per-step FreeActivations (device
        // copy re-uploads on demand; the host data stays authoritative).
        int imageContextLen = 0;
        Tensor encoderProj;
        if (_ctxCache.TryGetValue(encoder, out (Tensor Ctx, int ImageContextLen, Tensor? ImgKey) cached)
            && ReferenceEquals(cached.ImgKey, imageEmbeds))
        {
            encoderProj = cached.Ctx;
            imageContextLen = cached.ImageContextLen;
        }
        else
        {
            Tensor textProj = WanDitOps.TextEmbed(backend, encoder, dim, _textW1!, _textB1, _textW2!, _textB2);
            WanVideoDebugDump.Dump("cond_textProj", textProj);

            // I2V: project the CLIP image embeds and prepend them to the text context; the blocks split at imageContextLen.
            encoderProj = textProj;
            if (imageEmbeds is not null && _imgEmbedder.IsLoaded)
            {
                Tensor imgProj = _imgEmbedder.Forward(backend, imageEmbeds, dim);
                imageContextLen = (int)imgProj.Shape[0];
                encoderProj = WanDitOps.ConcatRows(imgProj, textProj, dim);
                imgProj.Dispose();
                textProj.Dispose();
            }
            _ = (nint)encoderProj.DataPointer;   // host-materialize (see cache note above)
            if (_ctxCache.Count >= 4)   // cap: prompts/images changed across gens; drop the stale contexts
            {
                foreach ((Tensor ctx, _, _) in _ctxCache.Values) ctx.Dispose();
                _ctxCache.Clear();
            }
            // Same-encoder re-entry with a different image replaces the stale entry (dispose the old context).
            if (_ctxCache.Remove(encoder, out (Tensor Ctx, int ImageContextLen, Tensor? ImgKey) stale)) stale.Ctx.Dispose();
            _ctxCache[encoder] = (encoderProj, imageContextLen, imageEmbeds);
        }

        Tensor cur = hidden;
        _fwdCounter++;
        string? vramLog = Environment.GetEnvironmentVariable("HARTSY_WAN_VRAM");
        for (int i = 0; i < _blocks.Length; i++)
        {
            if (vramLog is not null)
                System.IO.File.AppendAllText(vramLog, $"fwd#{_fwdCounter} block {i}: free {backend.FreeMemoryBytes() / (1024.0 * 1024 * 1024):F3} GB\n");
            Tensor next = _blocks[i].Forward(backend, cur, encoderProj, timestepProj, _rope, cos, sin, tokensPerGroup,
                imageContextLen: imageContextLen, dbg: i == 0 ? "b0" : null);
            cur.Dispose();
            cur = next;
            WanVideoDebugDump.Dump($"blocks.{i}", cur);
        }
        timestepProj.Dispose();   // cos/sin/encoderProj live in the per-generation caches — not per-forward temporaries

        Tensor projected = WanDitOps.FinalLayer(backend, cur, temb, _finalScaleShift!, _projOutW!, _projOutB, s, dim, _config.Eps, tokensPerGroup);
        cur.Dispose();
        temb.Dispose();
        Tensor outVel = WanDitOps.Unpatchify(projected, _config.OutChannels, gt, gh, gw, _config.PatchSize);
        projected.Dispose();
        WanVideoDebugDump.DumpOutput(outVel);
        return outVel;
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
