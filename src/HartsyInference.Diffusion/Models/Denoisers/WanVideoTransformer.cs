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
    private Tensor? _projOutW, _projOutB;      // [out·pt·ph·pw, inner]
    private Tensor? _finalScaleShift;          // [2, inner]
    private Tensor? _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B;   // time_embedder
    private Tensor? _timeProjW, _timeProjB;    // → 6·inner
    private Tensor? _textW1, _textB1, _textW2, _textB2;   // text_embedder
    // I2V image embedder (condition_embedder.image_embedder): norm1 → ff(net.0.proj, net.2) → norm2, + pos_embed.
    private Tensor? _imgNorm1W, _imgNorm1B, _imgFf0W, _imgFf0B, _imgFf2W, _imgFf2B, _imgNorm2W, _imgNorm2B, _imgPosEmbed;

    public WanVideoTransformer(WanVideoConfig config)
    {
        _config = config;
        _blocks = new WanVideoBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++) _blocks[i] = new WanVideoBlock(config, crossAttnNorm: true);
        _rope = new WanRope(config.HeadDim, config.RopeTheta, config.RopeMaxSeqLen);
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
        if (w.TryGetValue("condition_embedder.image_embedder.norm1.weight", out Tensor? in1))
        {
            _imgNorm1W = LoadF32In(in1); _imgNorm1B = LoadF32Opt(w, "condition_embedder.image_embedder.norm1.bias");
            _imgFf0W = w["condition_embedder.image_embedder.ff.net.0.proj.weight"]; w.TryGetValue("condition_embedder.image_embedder.ff.net.0.proj.bias", out _imgFf0B);
            _imgFf2W = w["condition_embedder.image_embedder.ff.net.2.weight"]; w.TryGetValue("condition_embedder.image_embedder.ff.net.2.bias", out _imgFf2B);
            _imgNorm2W = LoadF32(w, "condition_embedder.image_embedder.norm2.weight"); _imgNorm2B = LoadF32Opt(w, "condition_embedder.image_embedder.norm2.bias");
            _imgPosEmbed = LoadF32Opt(w, "condition_embedder.image_embedder.pos_embed");
        }
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"blocks.{i}");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _patchW2d, _patchB, _projOutW, _projOutB, _finalScaleShift,
            _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _timeProjW, _timeProjB, _textW1, _textB1, _textW2, _textB2,
            _imgNorm1W, _imgNorm1B, _imgFf0W, _imgFf0B, _imgFf2W, _imgFf2B, _imgNorm2W, _imgNorm2B, _imgPosEmbed })
            if (t is not null) yield return t;
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

        (Tensor cos, Tensor sin) = _rope.BuildCosSin(gt, gh, gw);

        Tensor hidden = WanDitOps.Patchify(backend, latent, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchB);   // [S, dim]
        WanVideoDebugDump.Dump("patch_embed", hidden);

        (Tensor temb, Tensor timestepProj) = WanDitOps.ConditionTimeGroups(backend, timesteps, _config.FreqDim, dim,
            _timeEmb1W!, _timeEmb1B, _timeEmb2W!, _timeEmb2B, _timeProjW!, _timeProjB);   // [G, dim], [G, 6, dim]
        Tensor textProj = WanDitOps.TextEmbed(backend, encoder, dim, _textW1!, _textB1, _textW2!, _textB2);

        // I2V: project the CLIP image embeds and prepend them to the text context; the blocks split at imageContextLen.
        int imageContextLen = 0;
        Tensor encoderProj = textProj;
        if (imageEmbeds is not null && _imgFf0W is not null)
        {
            Tensor imgProj = ImageEmbed(backend, imageEmbeds, dim);
            imageContextLen = (int)imgProj.Shape[0];
            encoderProj = ConcatRows(imgProj, textProj, dim);
            imgProj.Dispose();
            textProj.Dispose();
        }

        Tensor cur = hidden;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, cur, encoderProj, timestepProj, _rope, cos, sin, tokensPerGroup,
                imageContextLen: imageContextLen);
            cur.Dispose();
            cur = next;
            WanVideoDebugDump.Dump($"blocks.{i}", cur);
        }
        cos.Dispose(); sin.Dispose(); timestepProj.Dispose(); encoderProj.Dispose();

        Tensor projected = WanDitOps.FinalLayer(backend, cur, temb, _finalScaleShift!, _projOutW!, _projOutB, s, dim, _config.Eps, tokensPerGroup);
        cur.Dispose();
        temb.Dispose();
        Tensor outVel = WanDitOps.Unpatchify(projected, _config.OutChannels, gt, gh, gw, _config.PatchSize);
        projected.Dispose();
        WanVideoDebugDump.DumpOutput(outVel);
        return outVel;
    }

    /// <summary>Image embedder (<c>condition_embedder.image_embedder</c>): optional pos-embed add → FP32 LayerNorm →
    /// gelu FFN (net.0.proj → net.2) → FP32 LayerNorm. Maps CLIP <c>[seqImg, imageDim]</c> → <c>[seqImg, dim]</c>.</summary>
    private Tensor ImageEmbed(IBackend backend, Tensor imageEmbeds, int dim)
    {
        int seq = (int)imageEmbeds.Shape[0];
        int imageDim = (int)imageEmbeds.Shape[1];

        Tensor x = imageEmbeds;
        bool ownX = false;
        int peRank = _imgPosEmbed?.Shape.Rank ?? 0;
        if (_imgPosEmbed is not null && (int)_imgPosEmbed.Shape[peRank - 2] == seq && (int)_imgPosEmbed.Shape[peRank - 1] == imageDim)
        {
            x = new Tensor(new TensorShape(seq, imageDim), DType.F32);
            ownX = true;
            float* sp = (float*)imageEmbeds.DataPointer, pp = (float*)_imgPosEmbed.DataPointer, xp = (float*)x.DataPointer;
            long n = (long)seq * imageDim;
            for (long i = 0; i < n; i++) xp[i] = sp[i] + pp[i];
        }

        Tensor n1 = LayerNormAffine(x, _imgNorm1W!, _imgNorm1B, seq, imageDim);
        if (ownX) x.Dispose();
        int inner = (int)_imgFf0W!.Shape[0];
        Tensor h0 = new Tensor(new TensorShape(seq, inner), DType.F32); backend.Linear(h0, n1, _imgFf0W!, _imgFf0B); n1.Dispose();
        Tensor act = new Tensor(h0.Shape, DType.F32); backend.Gelu(act, h0); h0.Dispose();
        Tensor h2 = new Tensor(new TensorShape(seq, dim), DType.F32); backend.Linear(h2, act, _imgFf2W!, _imgFf2B); act.Dispose();
        Tensor outT = LayerNormAffine(h2, _imgNorm2W!, _imgNorm2B, seq, dim); h2.Dispose();
        return outT;
    }

    /// <summary>FP32 affine LayerNorm over the last dim.</summary>
    private Tensor LayerNormAffine(Tensor x, Tensor weight, Tensor? bias, int s, int d)
    {
        Tensor o = new Tensor(new TensorShape(s, d), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer, wp = (float*)weight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;
        for (int i = 0; i < s; i++)
        {
            long off = (long)i * d;
            double mean = 0; for (int k = 0; k < d; k++) mean += xp[off + k]; mean /= d;
            double var = 0; for (int k = 0; k < d; k++) { double dd = xp[off + k] - mean; var += dd * dd; }
            float inv = 1f / MathF.Sqrt((float)(var / d) + _config.Eps);
            for (int k = 0; k < d; k++)
                op[off + k] = (float)((xp[off + k] - mean) * inv) * wp[k] + (bp is null ? 0f : bp[k]);
        }
        return o;
    }

    /// <summary>Row-concatenates <paramref name="top"/> <c>[a, dim]</c> over <paramref name="bottom"/> <c>[b, dim]</c> → <c>[a+b, dim]</c>.</summary>
    private static Tensor ConcatRows(Tensor top, Tensor bottom, int dim)
    {
        int a = (int)top.Shape[0], b = (int)bottom.Shape[0];
        Tensor o = new Tensor(new TensorShape(a + b, dim), DType.F32);
        Buffer.MemoryCopy((float*)top.DataPointer, (float*)o.DataPointer, (long)a * dim * 4, (long)a * dim * 4);
        Buffer.MemoryCopy((float*)bottom.DataPointer, (float*)o.DataPointer + (long)a * dim, (long)b * dim * 4, (long)b * dim * 4);
        return o;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key) { Tensor t = w[key]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }
    private static Tensor LoadF32In(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    private static Tensor? LoadF32Opt(IReadOnlyDictionary<string, Tensor> w, string key) => w.TryGetValue(key, out Tensor? t) ? (t.DType == DType.F32 ? t : t.CastTo(DType.F32)) : null;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _patchW2d = _patchB = _projOutW = _projOutB = _finalScaleShift = null;
            _timeEmb1W = _timeEmb1B = _timeEmb2W = _timeEmb2B = _timeProjW = _timeProjB = null;
            _textW1 = _textB1 = _textW2 = _textB2 = null;
            _imgNorm1W = _imgNorm1B = _imgFf0W = _imgFf0B = _imgFf2W = _imgFf2B = _imgNorm2W = _imgNorm2B = _imgPosEmbed = null;
        }
        GC.SuppressFinalize(this);
    }
}
