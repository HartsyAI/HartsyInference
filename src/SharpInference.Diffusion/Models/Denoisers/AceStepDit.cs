using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Music;

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>ACE-Step v1 DiT (<c>ACEStepTransformer2DModel</c>) — 24 PixArt-style blocks over a 1-D time-token stream:
/// per-block 6-way AdaLN from a global <c>t_block</c> projection, <b>LiteLA ReLU-linear self-attention</b> (the Sana
/// kernel — confirmed from <c>CustomLiteLAProcessor2_0</c>, NOT softmax) with 1-D RoPE θ=1e6, softmax cross-attention
/// over the concatenated [speaker(1) ‖ text ‖ lyric] context (RoPE on the audio-positioned queries), and a GLUMBConv
/// FFN. Owns the conditioning projections: <c>speaker_embedder</c> (512→2560), <c>genre_embedder</c> (768→2560),
/// the lyric embedding + <see cref="AceStepLyricEncoder"/> + <c>lyric_proj</c> (1024→2560). Patch embed collapses the
/// 16-tall mel-latent height; the final layer restores it. Key spellings carry documented fallbacks until a real
/// checkpoint dump (validation-gated). Numerics validation-pending.</summary>
public sealed unsafe class AceStepDit : IDisposable
{
    private readonly AceStepConfig _config;
    private readonly AceStepLyricEncoder _lyricEncoder;
    private Block[] _blocks = [];
    private int _disposed;

    private Tensor? _projInConv1W, _projInConv1B, _projInNormW, _projInNormB, _projInConv2W, _projInConv2B;
    private Tensor? _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _tBlockW, _tBlockB;
    private Tensor? _speakerW, _speakerB, _genreW, _genreB, _lyricEmbs, _lyricProjW, _lyricProjB;
    private Tensor? _finalScaleShift, _projOutW, _projOutB;

    public AceStepDit(AceStepConfig config)
    {
        _config = config;
        _lyricEncoder = new AceStepLyricEncoder(config.LyricHiddenDim, numLayers: config.LyricEncoderLayers);
    }

    public AceStepConfig Config => _config;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _projInConv1W = Pick(w, "proj_in.conv1.weight", "proj_in.early_conv_layers.0.weight");
        _projInConv1B = PickOpt(w, "proj_in.conv1.bias", "proj_in.early_conv_layers.0.bias");
        _projInNormW = PickF32(w, "proj_in.norm.weight", "proj_in.early_conv_layers.1.weight");
        _projInNormB = PickOpt(w, "proj_in.norm.bias", "proj_in.early_conv_layers.1.bias");
        _projInConv2W = Pick(w, "proj_in.conv2.weight", "proj_in.early_conv_layers.2.weight");
        _projInConv2B = PickOpt(w, "proj_in.conv2.bias", "proj_in.early_conv_layers.2.bias");

        _timeEmb1W = w["time_embed.timestep_embedder.linear_1.weight"]; w.TryGetValue("time_embed.timestep_embedder.linear_1.bias", out _timeEmb1B);
        _timeEmb2W = w["time_embed.timestep_embedder.linear_2.weight"]; w.TryGetValue("time_embed.timestep_embedder.linear_2.bias", out _timeEmb2B);
        _tBlockW = Pick(w, "t_block.1.weight", "time_embed.linear.weight");
        _tBlockB = PickOpt(w, "t_block.1.bias", "time_embed.linear.bias");

        _speakerW = Pick(w, "speaker_embedder.weight", "speaker_embedder.linear.weight");
        _speakerB = PickOpt(w, "speaker_embedder.bias", "speaker_embedder.linear.bias");
        _genreW = Pick(w, "genre_embedder.weight", "caption_projection.linear.weight", "caption_projection.weight");
        _genreB = PickOpt(w, "genre_embedder.bias", "caption_projection.linear.bias", "caption_projection.bias");
        _lyricEmbs = w["lyric_embs.weight"];
        _lyricProjW = Pick(w, "lyric_proj.weight", "lyric_proj.linear.weight");
        _lyricProjB = PickOpt(w, "lyric_proj.bias", "lyric_proj.linear.bias");
        _lyricEncoder.LoadWeights(w, "lyric_encoder");

        _blocks = new Block[_config.NumLayers];
        for (int i = 0; i < _config.NumLayers; i++)
        {
            _blocks[i] = new Block(_config);
            _blocks[i].LoadWeights(w, $"transformer_blocks.{i}");
        }

        _finalScaleShift = PickF32(w, "proj_out.scale_shift_table", "final_layer.scale_shift_table", "scale_shift_table");
        _projOutW = Pick(w, "proj_out.linear.weight", "final_layer.linear.weight", "proj_out.weight");
        _projOutB = PickOpt(w, "proj_out.linear.bias", "final_layer.linear.bias", "proj_out.bias");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _projInConv1W, _projInConv1B, _projInNormW, _projInNormB, _projInConv2W, _projInConv2B,
            _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _tBlockW, _tBlockB,
            _speakerW, _speakerB, _genreW, _genreB, _lyricEmbs, _lyricProjW, _lyricProjB,
            _finalScaleShift, _projOutW, _projOutB })
            if (t is not null) yield return t;
        foreach (Tensor t in _lyricEncoder.EnumerateWeights()) yield return t;
        foreach (Block b in _blocks)
            foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Builds the cross-attention context <c>[1 + T_text + T_lyric, 2560]</c>: projected speaker token (zero
    /// vector when absent), projected UMT5 features <c>[T_text, 768]</c>, and the Conformer-encoded projected lyric
    /// token ids. Pass an empty <paramref name="lyricIds"/> for instrumental-only via tags in the prompt.</summary>
    public Tensor BuildContext(IBackend backend, Tensor textEmbeds, ReadOnlySpan<int> lyricIds, float[]? speakerVec)
    {
        int dim = _config.InnerDim;
        int tText = (int)textEmbeds.Shape[0];
        int tLyric = lyricIds.Length;

        Tensor ctx = new Tensor(new TensorShape(1 + tText + tLyric, dim), DType.F32);
        float* cp = (float*)ctx.DataPointer;

        // Speaker token.
        Tensor spk = new Tensor(new TensorShape(1, _config.SpeakerDim), DType.F32);
        if (speakerVec is not null)
        {
            if (speakerVec.Length != _config.SpeakerDim)
                throw new ArgumentException($"speaker vector must be {_config.SpeakerDim}-d.", nameof(speakerVec));
            speakerVec.CopyTo(new Span<float>((float*)spk.DataPointer, _config.SpeakerDim));
        }
        Tensor spkTok = new Tensor(new TensorShape(1, dim), DType.F32);
        backend.Linear(spkTok, spk, _speakerW!, _speakerB);
        spk.Dispose();
        Buffer.MemoryCopy((float*)spkTok.DataPointer, cp, dim * 4, dim * 4);
        spkTok.Dispose();

        // Text tokens.
        if (tText > 0)
        {
            Tensor txt = new Tensor(new TensorShape(tText, dim), DType.F32);
            backend.Linear(txt, textEmbeds, _genreW!, _genreB);
            Buffer.MemoryCopy((float*)txt.DataPointer, cp + (long)dim, (long)tText * dim * 4, (long)tText * dim * 4);
            txt.Dispose();
        }

        // Lyric tokens: embed → Conformer → proj.
        if (tLyric > 0)
        {
            int lh = _config.LyricHiddenDim;
            Tensor emb = new Tensor(new TensorShape(tLyric, lh), DType.F32);
            float* ep = (float*)emb.DataPointer;
            float* tablePtr = (float*)_lyricEmbs!.DataPointer;
            for (int i = 0; i < tLyric; i++)
            {
                int id = lyricIds[i];
                if (id < 0 || id >= _config.LyricVocabSize)
                    throw new ArgumentException($"lyric token {id} out of range.", nameof(lyricIds));
                Buffer.MemoryCopy(tablePtr + (long)id * lh, ep + (long)i * lh, lh * 4, lh * 4);
            }
            Tensor encoded = _lyricEncoder.Forward(backend, emb);
            emb.Dispose();
            Tensor projected = new Tensor(new TensorShape(tLyric, dim), DType.F32);
            backend.Linear(projected, encoded, _lyricProjW!, _lyricProjB);
            encoded.Dispose();
            Buffer.MemoryCopy((float*)projected.DataPointer, cp + (long)(1 + tText) * dim,
                (long)tLyric * dim * 4, (long)tLyric * dim * 4);
            projected.Dispose();
        }
        return ctx;
    }

    /// <summary>Velocity prediction: latent <c>[1, 8, 16, F]</c> + context (from <see cref="BuildContext"/>) +
    /// timestep (σ·1000) → <c>[1, 8, 16, F]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor context, float timestep)
    {
        int f = (int)latent.Shape[3];
        int dim = _config.InnerDim;
        if ((int)latent.Shape[2] != _config.LatentHeight)
            throw new ArgumentException($"latent height must be {_config.LatentHeight}.", nameof(latent));

        Tensor tokens = PatchEmbed(backend, latent, f);
        (Tensor temb, Tensor tBlock) = TimeEmbed(backend, timestep);
        (float[] cos, float[] sin) = BuildRope(f);
        (float[] cosCross, float[] sinCross) = BuildRope((int)context.Shape[0]);

        Tensor cur = tokens;
        foreach (Block b in _blocks)
        {
            Tensor next = b.Forward(backend, cur, context, tBlock, cos, sin, cosCross, sinCross, f);
            cur.Dispose();
            cur = next;
        }

        Tensor velocity = FinalLayer(backend, cur, temb, f);
        cur.Dispose();
        temb.Dispose();
        tBlock.Dispose();
        return velocity;
    }

    private Tensor PatchEmbed(IBackend backend, Tensor latent, int f)
    {
        int hid = _config.PatchEmbedHidden, h = _config.LatentHeight, dim = _config.InnerDim;
        Tensor c1 = new Tensor(new TensorShape(1, hid, h, f), DType.F32);
        backend.Conv2D(c1, latent, _projInConv1W!, _projInConv1B, 1, 1, 1, 1);
        Tensor n = new Tensor(c1.Shape, DType.F32);
        backend.GroupNorm(n, c1, _projInNormW!, _projInNormB!, Math.Min(32, hid), 1e-5f);
        c1.Dispose();
        // Height-collapsing conv: kernel (16, 1), stride (16, 1) → [1, dim, 1, F].
        Tensor c2 = new Tensor(new TensorShape(1, dim, 1, f), DType.F32);
        backend.Conv2D(c2, n, _projInConv2W!, _projInConv2B, _config.PatchSize.H, _config.PatchSize.W, 0, 0);
        n.Dispose();
        // → tokens [F, dim].
        Tensor tokens = new Tensor(new TensorShape(f, dim), DType.F32);
        float* sp = (float*)c2.DataPointer;
        float* tp = (float*)tokens.DataPointer;
        for (int d = 0; d < dim; d++)
            for (int i = 0; i < f; i++)
                tp[(long)i * dim + d] = sp[(long)d * f + i];
        c2.Dispose();
        return tokens;
    }

    private (Tensor Temb, Tensor TBlock) TimeEmbed(IBackend backend, float timestep)
    {
        int freq = _config.FreqDim, dim = _config.InnerDim;
        Tensor sinT = new Tensor(new TensorShape(1, freq), DType.F32);
        float* sp = (float*)sinT.DataPointer;
        int half = freq / 2;
        for (int i = 0; i < half; i++)
        {
            double f = Math.Exp(-Math.Log(10000.0) * i / half);
            sp[i] = (float)Math.Cos(timestep * f);
            sp[half + i] = (float)Math.Sin(timestep * f);
        }
        Tensor h1 = new Tensor(new TensorShape(1, dim), DType.F32);
        backend.Linear(h1, sinT, _timeEmb1W!, _timeEmb1B);
        sinT.Dispose();
        backend.Silu(h1, h1);
        Tensor temb = new Tensor(new TensorShape(1, dim), DType.F32);
        backend.Linear(temb, h1, _timeEmb2W!, _timeEmb2B);
        h1.Dispose();

        Tensor act = new Tensor(temb.Shape, DType.F32);
        backend.Silu(act, temb);
        Tensor tBlock = new Tensor(new TensorShape(1, 6 * dim), DType.F32);
        backend.Linear(tBlock, act, _tBlockW!, _tBlockB);
        act.Dispose();
        return (temb, tBlock);
    }

    private (float[] Cos, float[] Sin) BuildRope(int length)
    {
        int half = _config.HeadDim / 2;
        float[] cos = new float[length * half];
        float[] sin = new float[length * half];
        for (int pos = 0; pos < length; pos++)
            for (int i = 0; i < half; i++)
            {
                double freq = Math.Pow(_config.RopeTheta, -2.0 * i / _config.HeadDim);
                cos[pos * half + i] = (float)Math.Cos(pos * freq);
                sin[pos * half + i] = (float)Math.Sin(pos * freq);
            }
        return (cos, sin);
    }

    private Tensor FinalLayer(IBackend backend, Tensor tokens, Tensor temb, int f)
    {
        int dim = _config.InnerDim, h = _config.LatentHeight, c = _config.InChannels;
        float* ssp = (float*)_finalScaleShift!.DataPointer;
        float* tp = (float*)temb.DataPointer;
        Tensor normed = new Tensor(tokens.Shape, DType.F32);
        float* np = (float*)normed.DataPointer;
        float* xp = (float*)tokens.DataPointer;
        for (int i = 0; i < f; i++)
        {
            long off = (long)i * dim;
            double sum = 0;
            for (int d = 0; d < dim; d++) sum += (double)xp[off + d] * xp[off + d];
            float inv = 1f / MathF.Sqrt((float)(sum / dim) + 1e-6f);
            for (int d = 0; d < dim; d++)
            {
                float shift = ssp[d] + tp[d];
                float scale = ssp[dim + d] + tp[d];
                np[off + d] = xp[off + d] * inv * (1f + scale) + shift;
            }
        }
        int patchVec = h * c;
        Tensor projected = new Tensor(new TensorShape(f, patchVec), DType.F32);
        backend.Linear(projected, normed, _projOutW!, _projOutB);
        normed.Dispose();

        Tensor outT = new Tensor(new TensorShape([1L, c, h, f]), DType.F32);
        float* pp = (float*)projected.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int i = 0; i < f; i++)
            for (int p = 0; p < h; p++)
                for (int ci = 0; ci < c; ci++)
                    op[((long)ci * h + p) * f + i] = pp[(long)i * patchVec + p * c + ci];
        projected.Dispose();
        return outT;
    }

    private static Tensor Pick(IReadOnlyDictionary<string, Tensor> w, params string[] keys)
    {
        foreach (string k in keys)
            if (w.TryGetValue(k, out Tensor? t)) return t;
        throw new KeyNotFoundException($"none of [{string.Join(", ", keys)}] found in checkpoint.");
    }

    private static Tensor? PickOpt(IReadOnlyDictionary<string, Tensor> w, params string[] keys)
    {
        foreach (string k in keys)
            if (w.TryGetValue(k, out Tensor? t)) return t;
        return null;
    }

    private static Tensor PickF32(IReadOnlyDictionary<string, Tensor> w, params string[] keys)
    {
        Tensor t = Pick(w, keys);
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0) _blocks = [];
        GC.SuppressFinalize(this);
    }

    /// <summary>One LinearTransformerBlock: AdaLN-6 → LiteLA self-attn (gated) → softmax cross-attn → AdaLN'd
    /// GLUMBConv FFN (gated).</summary>
    private sealed class Block
    {
        private readonly AceStepConfig _c;
        private Tensor? _scaleShift;
        private Tensor? _qW, _kW, _vW, _oW, _oB, _normQ, _normK;
        private Tensor? _cqW, _ckW, _cvW, _coW, _coB, _cNormQ, _cNormK;
        private Tensor? _ffInvW, _ffInvB, _ffDepthW, _ffDepthB, _ffPointW, _ffPointB;

        public Block(AceStepConfig c) => _c = c;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _scaleShift = PickF32(w, $"{p}.scale_shift_table");
            _qW = Pick(w, $"{p}.attn.to_q.weight", $"{p}.attn1.to_q.weight");
            _kW = Pick(w, $"{p}.attn.to_k.weight", $"{p}.attn1.to_k.weight");
            _vW = Pick(w, $"{p}.attn.to_v.weight", $"{p}.attn1.to_v.weight");
            _oW = Pick(w, $"{p}.attn.to_out.0.weight", $"{p}.attn1.to_out.0.weight");
            _oB = PickOpt(w, $"{p}.attn.to_out.0.bias", $"{p}.attn1.to_out.0.bias");
            _normQ = PickOptF32(w, $"{p}.attn.norm_q.weight", $"{p}.attn1.norm_q.weight");
            _normK = PickOptF32(w, $"{p}.attn.norm_k.weight", $"{p}.attn1.norm_k.weight");

            _cqW = Pick(w, $"{p}.cross_attn.to_q.weight", $"{p}.attn2.to_q.weight");
            _ckW = Pick(w, $"{p}.cross_attn.to_k.weight", $"{p}.attn2.to_k.weight");
            _cvW = Pick(w, $"{p}.cross_attn.to_v.weight", $"{p}.attn2.to_v.weight");
            _coW = Pick(w, $"{p}.cross_attn.to_out.0.weight", $"{p}.attn2.to_out.0.weight");
            _coB = PickOpt(w, $"{p}.cross_attn.to_out.0.bias", $"{p}.attn2.to_out.0.bias");
            _cNormQ = PickOptF32(w, $"{p}.cross_attn.norm_q.weight", $"{p}.attn2.norm_q.weight");
            _cNormK = PickOptF32(w, $"{p}.cross_attn.norm_k.weight", $"{p}.attn2.norm_k.weight");

            _ffInvW = Squeeze1d(Pick(w, $"{p}.ff.inverted_conv.conv.weight", $"{p}.ff.conv_inverted.weight"));
            _ffInvB = PickOpt(w, $"{p}.ff.inverted_conv.conv.bias", $"{p}.ff.conv_inverted.bias");
            _ffDepthW = Squeeze1d(Pick(w, $"{p}.ff.depth_conv.conv.weight", $"{p}.ff.conv_depth.weight"));
            _ffDepthB = PickOpt(w, $"{p}.ff.depth_conv.conv.bias", $"{p}.ff.conv_depth.bias");
            _ffPointW = Squeeze1d(Pick(w, $"{p}.ff.point_conv.conv.weight", $"{p}.ff.conv_point.weight"));
            _ffPointB = PickOpt(w, $"{p}.ff.point_conv.conv.bias", $"{p}.ff.conv_point.bias");
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor? t in new[] { _scaleShift, _qW, _kW, _vW, _oW, _oB, _normQ, _normK,
                _cqW, _ckW, _cvW, _coW, _coB, _cNormQ, _cNormK,
                _ffInvW, _ffInvB, _ffDepthW, _ffDepthB, _ffPointW, _ffPointB })
                if (t is not null) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor x, Tensor ctx, Tensor tBlock,
            float[] cos, float[] sin, float[] cosCross, float[] sinCross, int f)
        {
            int dim = _c.InnerDim;
            float* ssp = (float*)_scaleShift!.DataPointer;
            float* tbp = (float*)tBlock.DataPointer;

            Tensor cur = Clone(x);

            // Self-attention with AdaLN modulation + gate.
            Tensor n1 = RmsNormNoAffine(cur, f, dim);
            ModulateInPlace(n1, ssp, tbp, dim, shiftIdx: 0, scaleIdx: 1, f);
            Tensor attn = LiteLaSelfAttention(backend, n1, cos, sin, f);
            n1.Dispose();
            GatedAdd(cur, attn, ssp, tbp, dim, gateIdx: 2, f);
            attn.Dispose();

            // Cross-attention (no modulation, no gate).
            Tensor cross = CrossAttention(backend, cur, ctx, cos, sin, cosCross, sinCross, f);
            AddInPlace(cur, cross);
            cross.Dispose();

            // FFN with AdaLN modulation + gate.
            Tensor n2 = RmsNormNoAffine(cur, f, dim);
            ModulateInPlace(n2, ssp, tbp, dim, shiftIdx: 3, scaleIdx: 4, f);
            Tensor ff = GlumbConv1d(backend, n2, f);
            n2.Dispose();
            GatedAdd(cur, ff, ssp, tbp, dim, gateIdx: 5, f);
            ff.Dispose();
            return cur;
        }

        /// <summary>Sana LiteLA: per head, relu(q)/relu(k) linear attention with the ones-row normalizer.</summary>
        private Tensor LiteLaSelfAttention(IBackend backend, Tensor n, float[] cos, float[] sin, int f)
        {
            int dim = _c.InnerDim, heads = _c.NumHeads, hd = _c.HeadDim;
            Tensor q = Proj(backend, n, _qW!, null, f, dim);
            Tensor k = Proj(backend, n, _kW!, null, f, dim);
            Tensor v = Proj(backend, n, _vW!, null, f, dim);
            if (_normQ is not null) RmsNormHeads(q, _normQ, f, heads, hd);
            if (_normK is not null) RmsNormHeads(k, _normK, f, heads, hd);
            ApplyRope(q, cos, sin, f, heads, hd);
            ApplyRope(k, cos, sin, f, heads, hd);

            Tensor outT = new Tensor(new TensorShape(f, dim), DType.F32);
            float* qp = (float*)q.DataPointer;
            float* kp = (float*)k.DataPointer;
            float* vp = (float*)v.DataPointer;
            float* op = (float*)outT.DataPointer;
            double[,] vk = new double[hd + 1, hd];
            double[] acc = new double[hd];
            for (int h = 0; h < heads; h++)
            {
                Array.Clear(vk);
                int hOff = h * hd;
                for (int i = 0; i < f; i++)
                {
                    long off = (long)i * dim + hOff;
                    for (int dk = 0; dk < hd; dk++)
                    {
                        float kv = MathF.Max(kp[off + dk], 0f);
                        if (kv == 0f) continue;
                        for (int dv = 0; dv < hd; dv++) vk[dv, dk] += (double)vp[off + dv] * kv;
                        vk[hd, dk] += kv;
                    }
                }
                for (int i = 0; i < f; i++)
                {
                    long off = (long)i * dim + hOff;
                    Array.Clear(acc);
                    double norm = 0;
                    for (int dk = 0; dk < hd; dk++)
                    {
                        float qv = MathF.Max(qp[off + dk], 0f);
                        if (qv == 0f) continue;
                        for (int dv = 0; dv < hd; dv++) acc[dv] += vk[dv, dk] * qv;
                        norm += vk[hd, dk] * qv;
                    }
                    float inv = (float)(1.0 / (norm + 1e-15));
                    for (int dv = 0; dv < hd; dv++) op[off + dv] = (float)(acc[dv] * inv);
                }
            }
            q.Dispose(); k.Dispose(); v.Dispose();

            Tensor projected = Proj(backend, outT, _oW!, _oB, f, dim);
            outT.Dispose();
            return projected;
        }

        private Tensor CrossAttention(IBackend backend, Tensor x, Tensor ctx,
            float[] cos, float[] sin, float[] cosCross, float[] sinCross, int f)
        {
            int dim = _c.InnerDim, heads = _c.NumHeads, hd = _c.HeadDim;
            int l = (int)ctx.Shape[0];
            Tensor q = Proj(backend, x, _cqW!, null, f, dim);
            Tensor k = Proj(backend, ctx, _ckW!, null, l, dim);
            Tensor v = Proj(backend, ctx, _cvW!, null, l, dim);
            if (_cNormQ is not null) RmsNormHeads(q, _cNormQ, f, heads, hd);
            if (_cNormK is not null) RmsNormHeads(k, _cNormK, l, heads, hd);
            // Queries rotate at audio positions; keys at context positions (validation-gated).
            ApplyRope(q, cos, sin, f, heads, hd);
            ApplyRope(k, cosCross, sinCross, l, heads, hd);

            Tensor qMh = ToMultiHead(q, f, heads, hd); q.Dispose();
            Tensor kMh = ToMultiHead(k, l, heads, hd); k.Dispose();
            Tensor vMh = ToMultiHead(v, l, heads, hd); v.Dispose();
            Tensor attn = new Tensor(new TensorShape(1, heads, f, hd), DType.F32);
            backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, null, 1f / MathF.Sqrt(hd));
            qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

            Tensor merged = FromMultiHead(attn, f, heads, hd);
            attn.Dispose();
            Tensor projected = Proj(backend, merged, _coW!, _coB, f, dim);
            merged.Dispose();
            return projected;
        }

        /// <summary>GLUMBConv on the time axis: 1×1 expand ×2 → SiLU → depthwise k3 → SiLU-gate chunk → 1×1.</summary>
        private Tensor GlumbConv1d(IBackend backend, Tensor n, int f)
        {
            int dim = _c.InnerDim;
            int hidden2 = (int)_ffInvW!.Shape[0];
            Tensor cf = RowsToChannels(n, dim, f);
            Tensor inv = new Tensor(new TensorShape(1, hidden2, f), DType.F32);
            backend.Conv1d(inv, cf, _ffInvW!, _ffInvB, 1, 0, 0, 1, 1);
            cf.Dispose();
            backend.Silu(inv, inv);
            int k = (int)_ffDepthW!.Shape[_ffDepthW.Shape.Rank - 1];
            Tensor depth = new Tensor(inv.Shape, DType.F32);
            backend.Conv1d(depth, inv, _ffDepthW!, _ffDepthB, 1, k / 2, k / 2, 1, hidden2);
            inv.Dispose();

            int hidden = hidden2 / 2;
            Tensor gated = new Tensor(new TensorShape(1, hidden, f), DType.F32);
            float* dp = (float*)depth.DataPointer;
            float* gp = (float*)gated.DataPointer;
            for (int c = 0; c < hidden; c++)
                for (int i = 0; i < f; i++)
                {
                    float val = dp[(long)c * f + i];
                    float gate = dp[(long)(hidden + c) * f + i];
                    gp[(long)c * f + i] = val * (gate / (1f + MathF.Exp(-gate)));
                }
            depth.Dispose();

            Tensor outCf = new Tensor(new TensorShape(1, dim, f), DType.F32);
            backend.Conv1d(outCf, gated, _ffPointW!, _ffPointB, 1, 0, 0, 1, 1);
            gated.Dispose();
            Tensor rows = ChannelsToRows(outCf, dim, f);
            outCf.Dispose();
            return rows;
        }

        /// <summary>Conv weights may arrive as [O, I, 1, K] (2-D convs over the height-1 grid) — reshape to [O, I, K]
        /// once at load (same memory layout, copied).</summary>
        private static Tensor Squeeze1d(Tensor w)
        {
            if (w.Shape.Rank == 3) return w;
            long o = w.Shape[0], i = w.Shape[1];
            long k = w.Shape.ElementCount / (o * i);
            Tensor t = new Tensor(new TensorShape(o, i, k), w.DType);
            long bytes = w.Shape.ElementCount * (w.DType == DType.F32 ? 4 : 2);
            Buffer.MemoryCopy((void*)w.DataPointer, (void*)t.DataPointer, bytes, bytes);
            return t;
        }

        private static void ModulateInPlace(Tensor n, float* table, float* tBlock, int dim, int shiftIdx, int scaleIdx, int rows)
        {
            float* np = (float*)n.DataPointer;
            for (int i = 0; i < rows; i++)
                for (int d = 0; d < dim; d++)
                {
                    float shift = table[shiftIdx * dim + d] + tBlock[shiftIdx * dim + d];
                    float scale = table[scaleIdx * dim + d] + tBlock[scaleIdx * dim + d];
                    long off = (long)i * dim + d;
                    np[off] = np[off] * (1f + scale) + shift;
                }
        }

        private static void GatedAdd(Tensor target, Tensor value, float* table, float* tBlock, int dim, int gateIdx, int rows)
        {
            float* tp = (float*)target.DataPointer;
            float* vp = (float*)value.DataPointer;
            for (int i = 0; i < rows; i++)
                for (int d = 0; d < dim; d++)
                {
                    float gate = table[gateIdx * dim + d] + tBlock[gateIdx * dim + d];
                    tp[(long)i * dim + d] += gate * vp[(long)i * dim + d];
                }
        }

        private static Tensor RmsNormNoAffine(Tensor x, int rows, int dim)
        {
            Tensor o = new Tensor(new TensorShape(rows, dim), DType.F32);
            float* xp = (float*)x.DataPointer;
            float* op = (float*)o.DataPointer;
            for (int i = 0; i < rows; i++)
            {
                long off = (long)i * dim;
                double sum = 0;
                for (int d = 0; d < dim; d++) sum += (double)xp[off + d] * xp[off + d];
                float inv = 1f / MathF.Sqrt((float)(sum / dim) + 1e-6f);
                for (int d = 0; d < dim; d++) op[off + d] = xp[off + d] * inv;
            }
            return o;
        }

        private static void RmsNormHeads(Tensor x, Tensor weight, int rows, int heads, int hd)
        {
            float* xp = (float*)x.DataPointer;
            float* wp = (float*)weight.DataPointer;
            int dim = heads * hd;
            for (int i = 0; i < rows; i++)
                for (int h = 0; h < heads; h++)
                {
                    long off = (long)i * dim + h * hd;
                    double sum = 0;
                    for (int d = 0; d < hd; d++) sum += (double)xp[off + d] * xp[off + d];
                    float inv = 1f / MathF.Sqrt((float)(sum / hd) + 1e-6f);
                    for (int d = 0; d < hd; d++) xp[off + d] = xp[off + d] * inv * wp[d];
                }
        }

        /// <summary>Interleaved-pair rotation: <c>out = x·cos + rot(x)·sin</c> with <c>rot = (−x₂ᵢ₊₁, x₂ᵢ)</c>.</summary>
        private static void ApplyRope(Tensor x, float[] cos, float[] sin, int rows, int heads, int hd)
        {
            float* xp = (float*)x.DataPointer;
            int dim = heads * hd, half = hd / 2;
            for (int i = 0; i < rows; i++)
                for (int h = 0; h < heads; h++)
                {
                    long off = (long)i * dim + h * hd;
                    for (int d = 0; d < half; d++)
                    {
                        float c = cos[i * half + d], s = sin[i * half + d];
                        float a = xp[off + 2 * d], b = xp[off + 2 * d + 1];
                        xp[off + 2 * d] = a * c - b * s;
                        xp[off + 2 * d + 1] = b * c + a * s;
                    }
                }
        }

        private static Tensor ToMultiHead(Tensor x, int rows, int heads, int hd)
        {
            Tensor o = new Tensor(new TensorShape(1, heads, rows, hd), DType.F32);
            float* xp = (float*)x.DataPointer;
            float* op = (float*)o.DataPointer;
            int dim = heads * hd;
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < rows; i++)
                    Buffer.MemoryCopy(xp + (long)i * dim + h * hd, op + ((long)h * rows + i) * hd, hd * 4, hd * 4);
            return o;
        }

        private static Tensor FromMultiHead(Tensor x, int rows, int heads, int hd)
        {
            Tensor o = new Tensor(new TensorShape(rows, heads * hd), DType.F32);
            float* xp = (float*)x.DataPointer;
            float* op = (float*)o.DataPointer;
            int dim = heads * hd;
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < rows; i++)
                    Buffer.MemoryCopy(xp + ((long)h * rows + i) * hd, op + (long)i * dim + h * hd, hd * 4, hd * 4);
            return o;
        }

        private static Tensor RowsToChannels(Tensor rows, int c, int t)
        {
            Tensor o = new Tensor(new TensorShape(1, c, t), DType.F32);
            float* rp = (float*)rows.DataPointer;
            float* op = (float*)o.DataPointer;
            for (int ci = 0; ci < c; ci++)
                for (int i = 0; i < t; i++) op[(long)ci * t + i] = rp[(long)i * c + ci];
            return o;
        }

        private static Tensor ChannelsToRows(Tensor cf, int c, int t)
        {
            Tensor o = new Tensor(new TensorShape(t, c), DType.F32);
            float* sp = (float*)cf.DataPointer;
            float* op = (float*)o.DataPointer;
            for (int ci = 0; ci < c; ci++)
                for (int i = 0; i < t; i++) op[(long)i * c + ci] = sp[(long)ci * t + i];
            return o;
        }

        private static Tensor Proj(IBackend backend, Tensor x, Tensor w, Tensor? b, int rows, int dim)
        {
            Tensor o = new Tensor(new TensorShape(rows, dim), DType.F32);
            backend.Linear(o, x, w, b);
            return o;
        }

        private static Tensor Clone(Tensor x)
        {
            Tensor o = new Tensor(x.Shape, DType.F32);
            long n = x.Shape.ElementCount;
            Buffer.MemoryCopy((float*)x.DataPointer, (float*)o.DataPointer, n * 4, n * 4);
            return o;
        }

        private static void AddInPlace(Tensor target, Tensor value)
        {
            long n = target.Shape.ElementCount;
            float* tp = (float*)target.DataPointer;
            float* vp = (float*)value.DataPointer;
            for (long i = 0; i < n; i++) tp[i] += vp[i];
        }

        private static Tensor Pick(IReadOnlyDictionary<string, Tensor> w, params string[] keys)
        {
            foreach (string key in keys)
                if (w.TryGetValue(key, out Tensor? t)) return t;
            throw new KeyNotFoundException($"none of [{string.Join(", ", keys)}] found in checkpoint.");
        }

        private static Tensor? PickOpt(IReadOnlyDictionary<string, Tensor> w, params string[] keys)
        {
            foreach (string key in keys)
                if (w.TryGetValue(key, out Tensor? t)) return t;
            return null;
        }

        private static Tensor? PickOptF32(IReadOnlyDictionary<string, Tensor> w, params string[] keys)
        {
            Tensor? t = PickOpt(w, keys);
            return t is null || t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        }

        private static Tensor PickF32(IReadOnlyDictionary<string, Tensor> w, params string[] keys)
        {
            Tensor t = Pick(w, keys);
            return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        }
    }
}
