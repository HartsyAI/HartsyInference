using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Stable Audio Open DiT forward pass (<see cref="StableAudioDitConfig"/>) — verified against the real
/// <c>FastVideo/stable-audio-open-small-Diffusers</c> checkpoint (key layout) and the upstream
/// <c>stable_audio_tools.models.dit.DiffusionTransformer</c> / <c>transformer.ContinuousTransformer</c> source
/// (block math, prepend-global-token conditioning, GLU chunk order). Per block: bias-ful pre-LayerNorm →
/// packed-QKV self-attention (per-head QK-LayerNorm + partial split-half RoPE) → bias-ful cross-attend-norm →
/// cross-attention (separate Q / packed-KV, per-head QK-LayerNorm, no RoPE) → bias-ful ff-norm → GLU-SwiGLU FFN
/// (single packed <c>proj</c> chunked into value ‖ gate, <c>value * silu(gate)</c>). No per-block AdaLN — the
/// timestep + timing conditioning is fused into one token prepended to the latent sequence.</summary>
public sealed unsafe class StableAudioDit : IDisposable
{
    private readonly StableAudioDitConfig _config;
    private Block[] _blocks = [];
    private int _disposed;

    private Tensor? _preprocessConvW, _postprocessConvW, _timestepFeaturesW, _projInW, _projOutW;
    private Tensor? _condEmbed0W, _condEmbed2W, _globalEmbed0W, _globalEmbed2W;
    private Tensor? _timestepEmbed0W, _timestepEmbed0B, _timestepEmbed2W, _timestepEmbed2B;
    private float[]? _invFreq;

    public StableAudioDit(StableAudioDitConfig config) => _config = config;

    public StableAudioDitConfig Config => _config;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _preprocessConvW = Pick(w, "preprocess_conv.weight");
        _postprocessConvW = Pick(w, "postprocess_conv.weight");
        _timestepFeaturesW = PickF32(w, "timestep_features.weight");
        _condEmbed0W = Pick(w, "to_cond_embed.0.weight");
        _condEmbed2W = Pick(w, "to_cond_embed.2.weight");
        _globalEmbed0W = Pick(w, "to_global_embed.0.weight");
        _globalEmbed2W = Pick(w, "to_global_embed.2.weight");
        _timestepEmbed0W = Pick(w, "to_timestep_embed.0.weight");
        _timestepEmbed0B = PickOpt(w, "to_timestep_embed.0.bias");
        _timestepEmbed2W = Pick(w, "to_timestep_embed.2.weight");
        _timestepEmbed2B = PickOpt(w, "to_timestep_embed.2.bias");
        _projInW = Pick(w, "transformer.project_in.weight");
        _projOutW = Pick(w, "transformer.project_out.weight");

        Tensor invFreqT = PickF32(w, "transformer.rotary_pos_emb.inv_freq");
        _invFreq = new float[invFreqT.Shape.ElementCount];
        new ReadOnlySpan<float>((float*)invFreqT.DataPointer, _invFreq.Length).CopyTo(_invFreq);

        _blocks = new Block[_config.NumLayers];
        for (int i = 0; i < _config.NumLayers; i++)
        {
            _blocks[i] = new Block(_config);
            _blocks[i].LoadWeights(w, $"transformer.layers.{i}");
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _preprocessConvW, _postprocessConvW, _timestepFeaturesW, _projInW, _projOutW,
            _condEmbed0W, _condEmbed2W, _globalEmbed0W, _globalEmbed2W,
            _timestepEmbed0W, _timestepEmbed0B, _timestepEmbed2W, _timestepEmbed2B })
            if (t is not null) yield return t;
        foreach (Block b in _blocks)
            foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Velocity (rectified-flow) prediction. <paramref name="latent"/> is the Oobleck-VAE latent
    /// <c>[1, ioChannels, T]</c> (channels-first). <paramref name="condTokens"/> is the RAW (pre-<c>to_cond_embed</c>)
    /// cross-attention conditioning sequence <c>[1, Lc, condTokenDim]</c> — the caller concatenates the T5 prompt
    /// tokens with the timing Fourier token(s) before calling (<c>cross_attention_cond_ids</c> in the checkpoint's
    /// conditioner config). <paramref name="globalCond"/> is the RAW (pre-<c>to_global_embed</c>) global timing
    /// embedding <c>[1, 1, globalCondDim]</c> (<c>global_cond_ids</c> — the <c>seconds_total</c> Fourier embedding
    /// for Open Small), summed internally with the projected timestep embedding to form the single prepended
    /// global token. <paramref name="timestep"/> is the rectified-flow time in <c>[0,1]</c> (Open Small's sampler
    /// passes <c>sigma = t</c> directly — no v-prediction alpha/sigma transform).</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor condTokens, Tensor globalCond, float timestep)
    {
        int t = (int)latent.Shape[2];
        int io = _config.IoChannels, dim = _config.HiddenSize;

        Tensor pre = new Tensor(new TensorShape(1, io, t), DType.F32);
        backend.Conv1d(pre, latent, _preprocessConvW!, null, 1, 0, 0, 1, 1);
        backend.Add(pre, pre, latent);

        Tensor chLast = new Tensor(new TensorShape(1, t, io), DType.F32);
        backend.Transpose2D(chLast, pre, io, t);
        pre.Dispose();
        Tensor tokens = new Tensor(new TensorShape(1, t, dim), DType.F32);
        backend.Linear(tokens, chLast, _projInW!, null);
        chLast.Dispose();

        int lc = (int)condTokens.Shape[1];
        Tensor cond = MlpProj(backend, condTokens, _condEmbed0W!, _condEmbed2W!, lc, dim);
        Tensor globalProj = MlpProj(backend, globalCond, _globalEmbed0W!, _globalEmbed2W!, 1, dim);
        Tensor timeEmb = TimestepEmbed(backend, timestep, dim);
        backend.Add(globalProj, globalProj, timeEmb);
        timeEmb.Dispose();

        int rows = t + 1;
        Tensor hidden = new Tensor(new TensorShape(1, rows, dim), DType.F32);
        backend.Concat(hidden, [globalProj, tokens], 1);
        globalProj.Dispose();
        tokens.Dispose();

        (float[] cos, float[] sin) = StableAudioRope.BuildTables(_invFreq!, rows);

        Tensor cur = hidden;
        foreach (Block b in _blocks)
        {
            Tensor next = b.Forward(backend, cur, cond, rows, lc, cos, sin);
            cur.Dispose();
            cur = next;
        }
        cond.Dispose();

        Tensor projected = new Tensor(new TensorShape(1, rows, io), DType.F32);
        backend.Linear(projected, cur, _projOutW!, null);
        cur.Dispose();

        Tensor latentOnly = new Tensor(new TensorShape(1, t, io), DType.F32);
        backend.SliceRows(latentOnly, projected, 1);
        projected.Dispose();

        Tensor chFirst = new Tensor(new TensorShape(1, io, t), DType.F32);
        backend.Transpose2D(chFirst, latentOnly, t, io);
        latentOnly.Dispose();

        Tensor post = new Tensor(new TensorShape(1, io, t), DType.F32);
        backend.Conv1d(post, chFirst, _postprocessConvW!, null, 1, 0, 0, 1, 1);
        backend.Add(post, post, chFirst);
        chFirst.Dispose();
        return post;
    }

    /// <summary>The <c>to_cond_embed</c> / <c>to_global_embed</c> conditioning MLP: Linear(no bias) → SiLU →
    /// Linear(no bias), both stages projecting to <paramref name="dim"/>.</summary>
    private static Tensor MlpProj(IBackend backend, Tensor input, Tensor w0, Tensor w2, int rows, int dim)
    {
        Tensor h1 = new Tensor(new TensorShape(1, rows, dim), DType.F32);
        backend.Linear(h1, input, w0, null);
        backend.Silu(h1, h1);
        Tensor outp = new Tensor(new TensorShape(1, rows, dim), DType.F32);
        backend.Linear(outp, h1, w2, null);
        h1.Dispose();
        return outp;
    }

    /// <summary><c>timestep_features</c> (Gaussian Fourier, <c>cat(cos, sin)</c>) → <c>to_timestep_embed</c>
    /// (Linear+bias → SiLU → Linear+bias). Returns <c>[1, 1, dim]</c>.</summary>
    private Tensor TimestepEmbed(IBackend backend, float timestep, int dim)
    {
        int half = (int)_timestepFeaturesW!.Shape[0];
        Tensor feat = new Tensor(new TensorShape(1, 1, 2 * half), DType.F32);
        float* fp = (float*)feat.DataPointer;
        float* wp = (float*)_timestepFeaturesW!.DataPointer;
        for (int i = 0; i < half; i++)
        {
            float f = 2f * MathF.PI * timestep * wp[i];
            fp[i] = MathF.Cos(f);
            fp[half + i] = MathF.Sin(f);
        }
        Tensor h1 = new Tensor(new TensorShape(1, 1, dim), DType.F32);
        backend.Linear(h1, feat, _timestepEmbed0W!, _timestepEmbed0B);
        feat.Dispose();
        backend.Silu(h1, h1);
        Tensor temb = new Tensor(new TensorShape(1, 1, dim), DType.F32);
        backend.Linear(temb, h1, _timestepEmbed2W!, _timestepEmbed2B);
        h1.Dispose();
        return temb;
    }

    private static Tensor Pick(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? t) ? t : throw new KeyNotFoundException($"'{key}' not found in checkpoint.");

    private static Tensor? PickOpt(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? t) ? t : null;

    private static Tensor PickF32(IReadOnlyDictionary<string, Tensor> w, string key)
    {
        Tensor t = Pick(w, key);
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0) _blocks = [];
        GC.SuppressFinalize(this);
    }

    /// <summary>One <c>TransformerBlock</c>: plain residual (no AdaLN) self-attn → cross-attn → GLU-SwiGLU FFN.</summary>
    private sealed class Block
    {
        private readonly StableAudioDitConfig _c;
        private Tensor? _preNormW, _preNormB, _crossNormW, _crossNormB, _ffNormW, _ffNormB;
        private Tensor? _selfQkvW, _selfQNormW, _selfQNormB, _selfKNormW, _selfKNormB, _selfOutW;
        private Tensor? _crossQW, _crossKvW, _crossQNormW, _crossQNormB, _crossKNormW, _crossKNormB, _crossOutW;
        private Tensor? _ff0W, _ff0B, _ff2W, _ff2B;

        public Block(StableAudioDitConfig c) => _c = c;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _preNormW = PickF32(w, $"{p}.pre_norm.weight");
            _preNormB = PickF32(w, $"{p}.pre_norm.bias");
            _selfQkvW = Pick(w, $"{p}.self_attn.to_qkv.weight");
            _selfQNormW = PickF32(w, $"{p}.self_attn.q_norm.weight");
            _selfQNormB = PickF32(w, $"{p}.self_attn.q_norm.bias");
            _selfKNormW = PickF32(w, $"{p}.self_attn.k_norm.weight");
            _selfKNormB = PickF32(w, $"{p}.self_attn.k_norm.bias");
            _selfOutW = Pick(w, $"{p}.self_attn.to_out.weight");

            _crossNormW = PickF32(w, $"{p}.cross_attend_norm.weight");
            _crossNormB = PickF32(w, $"{p}.cross_attend_norm.bias");
            _crossQW = Pick(w, $"{p}.cross_attn.to_q.weight");
            _crossKvW = Pick(w, $"{p}.cross_attn.to_kv.weight");
            _crossQNormW = PickF32(w, $"{p}.cross_attn.q_norm.weight");
            _crossQNormB = PickF32(w, $"{p}.cross_attn.q_norm.bias");
            _crossKNormW = PickF32(w, $"{p}.cross_attn.k_norm.weight");
            _crossKNormB = PickF32(w, $"{p}.cross_attn.k_norm.bias");
            _crossOutW = Pick(w, $"{p}.cross_attn.to_out.weight");

            _ffNormW = PickF32(w, $"{p}.ff_norm.weight");
            _ffNormB = PickF32(w, $"{p}.ff_norm.bias");
            _ff0W = Pick(w, $"{p}.ff.ff.0.proj.weight");
            _ff0B = PickOpt(w, $"{p}.ff.ff.0.proj.bias");
            _ff2W = Pick(w, $"{p}.ff.ff.2.weight");
            _ff2B = PickOpt(w, $"{p}.ff.ff.2.bias");
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor? t in new[] { _preNormW, _preNormB, _selfQkvW, _selfQNormW, _selfQNormB, _selfKNormW,
                _selfKNormB, _selfOutW, _crossNormW, _crossNormB, _crossQW, _crossKvW, _crossQNormW, _crossQNormB,
                _crossKNormW, _crossKNormB, _crossOutW, _ffNormW, _ffNormB, _ff0W, _ff0B, _ff2W, _ff2B })
                if (t is not null) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor x, Tensor cond, int rows, int condLen, float[] cos, float[] sin)
        {
            int dim = _c.HiddenSize, heads = _c.NumHeads, hd = _c.HeadDim, rotDim = _c.RopeDim;

            Tensor n1 = new Tensor(x.Shape, DType.F32);
            backend.LayerNorm(n1, x, _preNormW!, _preNormB!, _c.LayerNormEps);
            Tensor selfOut = SelfAttention(backend, n1, rows, heads, hd, dim, rotDim, cos, sin);
            n1.Dispose();
            Tensor cur = new Tensor(x.Shape, DType.F32);
            backend.Add(cur, x, selfOut);
            selfOut.Dispose();

            Tensor n2 = new Tensor(cur.Shape, DType.F32);
            backend.LayerNorm(n2, cur, _crossNormW!, _crossNormB!, _c.LayerNormEps);
            Tensor crossOut = CrossAttention(backend, n2, cond, rows, condLen, heads, hd, dim);
            n2.Dispose();
            backend.Add(cur, cur, crossOut);
            crossOut.Dispose();

            Tensor n3 = new Tensor(cur.Shape, DType.F32);
            backend.LayerNorm(n3, cur, _ffNormW!, _ffNormB!, _c.LayerNormEps);
            Tensor ffOut = Ffn(backend, n3, rows, dim);
            n3.Dispose();
            backend.Add(cur, cur, ffOut);
            ffOut.Dispose();
            return cur;
        }

        private Tensor SelfAttention(IBackend backend, Tensor n, int rows, int heads, int hd, int dim, int rotDim,
            float[] cos, float[] sin)
        {
            Tensor qkv = new Tensor(new TensorShape(1, rows, 3 * dim), DType.F32);
            backend.Linear(qkv, n, _selfQkvW!, null);

            Tensor qFlat = new Tensor(new TensorShape(1, rows, dim), DType.F32);
            Tensor kFlat = new Tensor(new TensorShape(1, rows, dim), DType.F32);
            Tensor vFlat = new Tensor(new TensorShape(1, rows, dim), DType.F32);
            backend.SliceLastDim(qFlat, qkv, 0);
            backend.SliceLastDim(kFlat, qkv, dim);
            backend.SliceLastDim(vFlat, qkv, 2 * dim);
            qkv.Dispose();

            Tensor q = HeadsNormRope(backend, qFlat, rows, heads, hd, _selfQNormW!, _selfQNormB!, rotDim, cos, sin);
            Tensor k = HeadsNormRope(backend, kFlat, rows, heads, hd, _selfKNormW!, _selfKNormB!, rotDim, cos, sin);
            qFlat.Dispose(); kFlat.Dispose();
            Tensor v = new Tensor(new TensorShape(1, heads, rows, hd), DType.F32);
            ToMultiHead(v, vFlat, rows, heads, hd);
            vFlat.Dispose();

            Tensor attn = new Tensor(new TensorShape(1, heads, rows, hd), DType.F32);
            backend.ScaledDotProductAttention(attn, q, k, v, null, 1f / MathF.Sqrt(hd));
            q.Dispose(); k.Dispose(); v.Dispose();

            Tensor merged = new Tensor(new TensorShape(1, rows, dim), DType.F32);
            FromMultiHead(merged, attn, rows, heads, hd);
            attn.Dispose();
            Tensor outT = new Tensor(new TensorShape(1, rows, dim), DType.F32);
            backend.Linear(outT, merged, _selfOutW!, null);
            merged.Dispose();
            return outT;
        }

        private Tensor CrossAttention(IBackend backend, Tensor n, Tensor cond, int rows, int condLen, int heads,
            int hd, int dim)
        {
            int kvHeads = _c.CrossKvHeads, kvDim = kvHeads * hd;

            Tensor qFlat = new Tensor(new TensorShape(1, rows, dim), DType.F32);
            backend.Linear(qFlat, n, _crossQW!, null);
            Tensor q = HeadsNormRope(backend, qFlat, rows, heads, hd, _crossQNormW!, _crossQNormB!, 0, null, null);
            qFlat.Dispose();

            Tensor kv = new Tensor(new TensorShape(1, condLen, 2 * kvDim), DType.F32);
            backend.Linear(kv, cond, _crossKvW!, null);
            Tensor kFlat = new Tensor(new TensorShape(1, condLen, kvDim), DType.F32);
            Tensor vFlat = new Tensor(new TensorShape(1, condLen, kvDim), DType.F32);
            backend.SliceLastDim(kFlat, kv, 0);
            backend.SliceLastDim(vFlat, kv, kvDim);
            kv.Dispose();

            Tensor kHeads = HeadsNormRope(backend, kFlat, condLen, kvHeads, hd, _crossKNormW!, _crossKNormB!, 0, null, null);
            kFlat.Dispose();
            Tensor vHeads = new Tensor(new TensorShape(1, kvHeads, condLen, hd), DType.F32);
            ToMultiHead(vHeads, vFlat, condLen, kvHeads, hd);
            vFlat.Dispose();

            Tensor kRep, vRep;
            int group = heads / kvHeads;
            if (group == 1) { kRep = kHeads; vRep = vHeads; }
            else
            {
                kRep = new Tensor(new TensorShape(1, heads, condLen, hd), DType.F32);
                vRep = new Tensor(new TensorShape(1, heads, condLen, hd), DType.F32);
                RepeatKv(kRep, kHeads, kvHeads, group, condLen, hd);
                RepeatKv(vRep, vHeads, kvHeads, group, condLen, hd);
                kHeads.Dispose(); vHeads.Dispose();
            }

            Tensor attn = new Tensor(new TensorShape(1, heads, rows, hd), DType.F32);
            backend.ScaledDotProductAttention(attn, q, kRep, vRep, null, 1f / MathF.Sqrt(hd));
            q.Dispose(); kRep.Dispose(); vRep.Dispose();

            Tensor merged = new Tensor(new TensorShape(1, rows, dim), DType.F32);
            FromMultiHead(merged, attn, rows, heads, hd);
            attn.Dispose();
            Tensor outT = new Tensor(new TensorShape(1, rows, dim), DType.F32);
            backend.Linear(outT, merged, _crossOutW!, null);
            merged.Dispose();
            return outT;
        }

        /// <summary>flat [1,rows,H·D] → heads [1,H,rows,D] → per-head QK-LayerNorm → optional partial RoPE
        /// (pass <c>rotDim=0</c>/<c>cos=null</c> to skip, e.g. cross-attention).</summary>
        private Tensor HeadsNormRope(IBackend backend, Tensor flat, int rows, int heads, int hd,
            Tensor normW, Tensor normB, int rotDim, float[]? cos, float[]? sin)
        {
            Tensor headsT = new Tensor(new TensorShape(1, heads, rows, hd), DType.F32);
            ToMultiHead(headsT, flat, rows, heads, hd);
            Tensor normed = new Tensor(headsT.Shape, DType.F32);
            backend.LayerNorm(normed, headsT, normW, normB, _c.QkNormEps);
            headsT.Dispose();
            if (cos is not null) StableAudioRope.Apply(normed, heads, rows, hd, rotDim, cos, sin!);
            return normed;
        }

        /// <summary>GLU-SwiGLU FFN: single packed <c>ff.0.proj</c> chunked <c>[value ‖ gate]</c> along the last
        /// dim, <c>value * silu(gate)</c>, then <c>ff.2</c> projects back down (matches <c>stable_audio_tools</c>'s
        /// <c>GLU.forward</c>: <c>x, gate = proj(x).chunk(2, dim=-1); return x * act(gate)</c>).</summary>
        private Tensor Ffn(IBackend backend, Tensor n, int rows, int dim)
        {
            int ffInner = _c.FfInner;
            Tensor proj = new Tensor(new TensorShape(1, rows, 2 * ffInner), DType.F32);
            backend.Linear(proj, n, _ff0W!, _ff0B);

            Tensor val = new Tensor(new TensorShape(1, rows, ffInner), DType.F32);
            Tensor gate = new Tensor(new TensorShape(1, rows, ffInner), DType.F32);
            backend.SliceLastDim(val, proj, 0);
            backend.SliceLastDim(gate, proj, ffInner);
            proj.Dispose();

            backend.Silu(gate, gate);
            backend.Mul(val, val, gate);
            gate.Dispose();

            Tensor outT = new Tensor(new TensorShape(1, rows, dim), DType.F32);
            backend.Linear(outT, val, _ff2W!, _ff2B);
            val.Dispose();
            return outT;
        }

        /// <summary>[1, rows, H·D] → [1, H, rows, D].</summary>
        private static void ToMultiHead(Tensor outT, Tensor x, int rows, int heads, int hd)
        {
            float* xp = (float*)x.DataPointer;
            float* op = (float*)outT.DataPointer;
            int dim = heads * hd;
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < rows; i++)
                    Buffer.MemoryCopy(xp + (long)i * dim + h * hd, op + ((long)h * rows + i) * hd, hd * 4, hd * 4);
        }

        /// <summary>[1, H, rows, D] → [1, rows, H·D].</summary>
        private static void FromMultiHead(Tensor outT, Tensor x, int rows, int heads, int hd)
        {
            float* xp = (float*)x.DataPointer;
            float* op = (float*)outT.DataPointer;
            int dim = heads * hd;
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < rows; i++)
                    Buffer.MemoryCopy(xp + ((long)h * rows + i) * hd, op + (long)i * dim + h * hd, hd * 4, hd * 4);
        }

        /// <summary>[1, kvHeads, S, D] → [1, kvHeads·group, S, D] (GQA head replication for Open 1.0's 24Q:12KV
        /// cross-attention; Open Small is symmetric 8:8 so <c>group == 1</c> and this is never called).</summary>
        private static void RepeatKv(Tensor outT, Tensor input, int kvHeads, int group, int s, int hd)
        {
            float* ip = (float*)input.DataPointer;
            float* op = (float*)outT.DataPointer;
            long perHead = (long)s * hd;
            for (int h = 0; h < kvHeads; h++)
                for (int g = 0; g < group; g++)
                {
                    long srcOff = h * perHead;
                    long dstOff = (long)(h * group + g) * perHead;
                    Buffer.MemoryCopy(ip + srcOff, op + dstOff, perHead * 4, perHead * 4);
                }
        }

        private static Tensor Pick(IReadOnlyDictionary<string, Tensor> w, string key) =>
            w.TryGetValue(key, out Tensor? t) ? t : throw new KeyNotFoundException($"'{key}' not found in checkpoint.");

        private static Tensor? PickOpt(IReadOnlyDictionary<string, Tensor> w, string key) =>
            w.TryGetValue(key, out Tensor? t) ? t : null;

        private static Tensor PickF32(IReadOnlyDictionary<string, Tensor> w, string key)
        {
            Tensor t = Pick(w, key);
            return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        }
    }
}
