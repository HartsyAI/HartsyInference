using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Vits;

/// <summary>VITS text encoder (<c>enc_p</c>): phoneme embedding → N pre-norm transformer layers with
/// <b>relative-position</b> multi-head attention + Conv FFN → project to <c>2·inter</c> → split into the
/// prior mean/log-std <c>(m_p, logs_p)</c>. The relative-position term is computed directly (per query, a
/// clipped offset into the learned <c>emb_rel_k/v</c> tables) rather than via the pad-reshape rel↔abs
/// gymnastics — equivalent and far less error-prone, and cheap since the phoneme length is small.</summary>
public sealed unsafe class VitsTextEncoder
{
    private readonly VitsConfig _cfg;
    private readonly int _kCh;
    private Tensor? _emb, _projW, _projB;
    private readonly Layer[] _layers;

    public VitsTextEncoder(VitsConfig cfg)
    {
        _cfg = cfg;
        _kCh = cfg.HiddenChannels / cfg.NumHeads;
        _layers = new Layer[cfg.NumEncoderLayers];
        for (int i = 0; i < _layers.Length; i++) _layers[i] = new Layer(cfg, _kCh);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "enc_p")
    {
        _emb = VitsWeights.Conv(w, $"{prefix}.emb");     // [n_vocab, hidden] (no .weight_g; plain .weight)
        LoadWeightsLayersOnly(w, prefix);
    }

    /// <summary>Loads only the encoder layers + projection (no phoneme embedding) — used by MeloTTS, which
    /// supplies its own summed phoneme/tone/language/BERT embedding via <see cref="ForwardFromEmbedding"/>.</summary>
    public void LoadWeightsLayersOnly(IReadOnlyDictionary<string, Tensor> w, string prefix = "enc_p")
    {
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, $"{prefix}.encoder", i);
        _projW = VitsWeights.Conv(w, $"{prefix}.proj"); _projB = VitsWeights.Bias(w, $"{prefix}.proj");
    }

    /// <summary>Encodes phoneme ids → the encoder hidden <c>[1, hidden, T]</c> (consumed by the duration
    /// predictor) and the prior <c>(m_p, logs_p)</c>, each <c>[1, inter, T]</c> (channels-first).</summary>
    public (Tensor Hidden, Tensor MP, Tensor LogsP) Forward(IBackend backend, ReadOnlySpan<int> tokens)
    {
        int t = tokens.Length, h = _cfg.HiddenChannels;
        float scale = MathF.Sqrt(h);
        Tensor x = new(new TensorShape(1, h, t), DType.F32);     // channels-first
        float* xp = (float*)x.DataPointer;
        float* ep = (float*)_emb!.DataPointer;
        for (int j = 0; j < t; j++)
            for (int c = 0; c < h; c++) xp[(long)c * t + j] = ep[(long)tokens[j] * h + c] * scale;
        return ForwardFromEmbedding(backend, x, t);
    }

    /// <summary>Runs the encoder layers + projection over a prebuilt input embedding <c>[1, hidden, T]</c>
    /// (takes ownership). The MeloTTS entry point (caller sums phoneme + tone + language + BERT embeddings). When a
    /// speaker embedding <paramref name="g"/> <c>[1, gin, 1]</c> and the <paramref name="spkW"/>/<paramref name="spkB"/>
    /// <c>spk_emb_linear</c> weights are supplied, <c>x += spk_emb_linear(g)</c> is added before layer
    /// <paramref name="condLayerIdx"/> (VITS2 speaker-conditioned encoder).</summary>
    public (Tensor Hidden, Tensor MP, Tensor LogsP) ForwardFromEmbedding(IBackend backend, Tensor embed, int t,
        Tensor? g = null, Tensor? spkW = null, Tensor? spkB = null, int condLayerIdx = 2)
    {
        Tensor x = embed;
        float[]? gProj = null;
        if (g is not null && spkW is not null)
        {
            int h = _cfg.HiddenChannels, gin = _cfg.GinChannels;
            gProj = new float[h];
            float* gp = (float*)g.DataPointer, sw = (float*)spkW.DataPointer, sb = spkB is null ? null : (float*)spkB.DataPointer;
            for (int c = 0; c < h; c++)
            {
                float acc = sb is null ? 0f : sb[c];
                for (int k = 0; k < gin; k++) acc += sw[(long)c * gin + k] * gp[k];
                gProj[c] = acc;
            }
        }
        for (int i = 0; i < _layers.Length; i++)
        {
            if (gProj is not null && i == condLayerIdx)
            {
                float* xp = (float*)x.DataPointer;
                for (int c = 0; c < _cfg.HiddenChannels; c++)
                    for (int j = 0; j < t; j++) xp[(long)c * t + j] += gProj[c];
            }
            Tensor next = _layers[i].Forward(backend, x, t);
            x.Dispose(); x = next;
        }

        Tensor stats = new(new TensorShape(1, 2 * _cfg.InterChannels, t), DType.F32);
        backend.Conv1d(stats, x, _projW!, _projB, 1, 0, 0, 1, 1);
        int inter = _cfg.InterChannels;
        Tensor mP = new(new TensorShape(1, inter, t), DType.F32);
        Tensor logsP = new(new TensorShape(1, inter, t), DType.F32);
        float* sp = (float*)stats.DataPointer;
        Buffer.MemoryCopy(sp, (void*)mP.DataPointer, (long)inter * t * 4, (long)inter * t * 4);
        Buffer.MemoryCopy(sp + (long)inter * t, (void*)logsP.DataPointer, (long)inter * t * 4, (long)inter * t * 4);
        stats.Dispose();
        return (x, mP, logsP);     // caller owns x (the encoder hidden)
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_emb is not null) yield return _emb;
        foreach (Layer l in _layers) foreach (Tensor t in l.EnumerateWeights()) yield return t;
        if (_projW is not null) yield return _projW;
        if (_projB is not null) yield return _projB;
    }

    private sealed class Layer
    {
        private readonly VitsConfig _cfg;
        private readonly int _kCh, _w;
        private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _relK, _relV;
        private Tensor? _norm1G, _norm1B, _norm2G, _norm2B, _ffn1W, _ffn1B, _ffn2W, _ffn2B;

        public Layer(VitsConfig cfg, int kCh) { _cfg = cfg; _kCh = kCh; _w = cfg.WindowSize; }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, int i)
        {
            _qW = VitsWeights.Conv(w, $"{prefix}.attn_layers.{i}.conv_q"); _qB = VitsWeights.Bias(w, $"{prefix}.attn_layers.{i}.conv_q");
            _kW = VitsWeights.Conv(w, $"{prefix}.attn_layers.{i}.conv_k"); _kB = VitsWeights.Bias(w, $"{prefix}.attn_layers.{i}.conv_k");
            _vW = VitsWeights.Conv(w, $"{prefix}.attn_layers.{i}.conv_v"); _vB = VitsWeights.Bias(w, $"{prefix}.attn_layers.{i}.conv_v");
            _oW = VitsWeights.Conv(w, $"{prefix}.attn_layers.{i}.conv_o"); _oB = VitsWeights.Bias(w, $"{prefix}.attn_layers.{i}.conv_o");
            // rel-pos tables + LayerNorm gains/biases are read host-side as float* (Attention / AddNorm), so they
            // MUST be F32 — upcast an fp16 checkpoint's copies (a no-op when already F32). Without this an fp16 RVC/
            // MeloTTS checkpoint produced garbage/blank audio (fp16 bytes misread as f32).
            _relK = WhisperOps.EnsureF32(w[$"{prefix}.attn_layers.{i}.emb_rel_k"]); _relV = WhisperOps.EnsureF32(w[$"{prefix}.attn_layers.{i}.emb_rel_v"]);
            _norm1G = WhisperOps.EnsureF32(w[$"{prefix}.norm_layers_1.{i}.gamma"]); _norm1B = WhisperOps.EnsureF32(w[$"{prefix}.norm_layers_1.{i}.beta"]);
            _norm2G = WhisperOps.EnsureF32(w[$"{prefix}.norm_layers_2.{i}.gamma"]); _norm2B = WhisperOps.EnsureF32(w[$"{prefix}.norm_layers_2.{i}.beta"]);
            _ffn1W = VitsWeights.Conv(w, $"{prefix}.ffn_layers.{i}.conv_1"); _ffn1B = VitsWeights.Bias(w, $"{prefix}.ffn_layers.{i}.conv_1");
            _ffn2W = VitsWeights.Conv(w, $"{prefix}.ffn_layers.{i}.conv_2"); _ffn2B = VitsWeights.Bias(w, $"{prefix}.ffn_layers.{i}.conv_2");
        }

        public Tensor Forward(IBackend backend, Tensor x, int t)
        {
            int h = _cfg.HiddenChannels;
            Tensor attn = Attention(backend, x, t);
            Tensor afterAttn = AddNorm(backend, x, attn, _norm1G!, _norm1B!, h, t); attn.Dispose();

            // FFN: conv_1 (k3, same) → relu → conv_2 (k3, same).
            Tensor f1 = new(new TensorShape(1, _cfg.FilterChannels, t), DType.F32);
            backend.Conv1d(f1, afterAttn, _ffn1W!, _ffn1B, 1, 1, 1, 1, 1);
            float* fp = (float*)f1.DataPointer;
            for (long n = 0; n < f1.ElementCount; n++) if (fp[n] < 0) fp[n] = 0;
            Tensor f2 = new(new TensorShape(1, h, t), DType.F32);
            backend.Conv1d(f2, f1, _ffn2W!, _ffn2B, 1, 1, 1, 1, 1); f1.Dispose();
            Tensor outT = AddNorm(backend, afterAttn, f2, _norm2G!, _norm2B!, h, t);
            afterAttn.Dispose(); f2.Dispose();
            return outT;
        }

        private Tensor Attention(IBackend backend, Tensor x, int t)
        {
            int h = _cfg.HiddenChannels, heads = _cfg.NumHeads, kc = _kCh, w = _w;
            Tensor q = Proj(backend, x, _qW!, _qB, h, t);
            Tensor k = Proj(backend, x, _kW!, _kB, h, t);
            Tensor v = Proj(backend, x, _vW!, _vB, h, t);
            float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
            float* rk = (float*)_relK!.DataPointer, rv = (float*)_relV!.DataPointer;   // [1, 2w+1, kc]
            float invSqrt = 1f / MathF.Sqrt(kc);

            Tensor outT = new(new TensorShape(1, h, t), DType.F32);
            float* op = (float*)outT.DataPointer;
            float[] scores = new float[t];
            for (int hd = 0; hd < heads; hd++)
            {
                int chBase = hd * kc;
                for (int i = 0; i < t; i++)
                {
                    float maxS = float.NegativeInfinity;
                    for (int j = 0; j < t; j++)
                    {
                        // VITS uses learned relative embeddings only within ±window; positions outside the window are
                        // zero-padded (NOT clamped to the edge embedding).
                        int diff = j - i;
                        bool inWindow = diff >= -w && diff <= w;
                        int rel = diff + w;     // valid only when inWindow
                        float dot = 0, relDot = 0;
                        for (int c = 0; c < kc; c++)
                        {
                            float qv = qp[(long)(chBase + c) * t + i];
                            dot += qv * kp[(long)(chBase + c) * t + j];
                            if (inWindow) relDot += qv * rk[(long)rel * kc + c];
                        }
                        float s = (dot + relDot) * invSqrt;
                        scores[j] = s;
                        if (s > maxS) maxS = s;
                    }
                    float sum = 0;
                    for (int j = 0; j < t; j++) { scores[j] = MathF.Exp(scores[j] - maxS); sum += scores[j]; }
                    float inv = 1f / sum;
                    for (int c = 0; c < kc; c++)
                    {
                        float acc = 0;
                        for (int j = 0; j < t; j++)
                        {
                            float p = scores[j] * inv;
                            int diff = j - i;
                            bool inWindow = diff >= -w && diff <= w;
                            float relV = inWindow ? rv[(long)(diff + w) * kc + c] : 0f;
                            acc += p * (vp[(long)(chBase + c) * t + j] + relV);
                        }
                        op[(long)(chBase + c) * t + i] = acc;
                    }
                }
            }
            q.Dispose(); k.Dispose(); v.Dispose();
            Tensor proj = new(new TensorShape(1, h, t), DType.F32);
            backend.Conv1d(proj, outT, _oW!, _oB, 1, 0, 0, 1, 1);
            outT.Dispose();
            return proj;
        }

        private static Tensor Proj(IBackend backend, Tensor x, Tensor w, Tensor? b, int h, int t)
        {
            Tensor o = new(new TensorShape(1, h, t), DType.F32);
            backend.Conv1d(o, x, w, b, 1, 0, 0, 1, 1);
            return o;
        }

        /// <summary>Residual add + channel-dim LayerNorm (VITS normalizes over channels per time step).</summary>
        private Tensor AddNorm(IBackend backend, Tensor x, Tensor delta, Tensor gamma, Tensor beta, int h, int t)
        {
            Tensor outT = new(new TensorShape(1, h, t), DType.F32);
            float* xp = (float*)x.DataPointer, dp = (float*)delta.DataPointer, op = (float*)outT.DataPointer;
            float* g = (float*)gamma.DataPointer, be = (float*)beta.DataPointer;
            for (int j = 0; j < t; j++)
            {
                double mean = 0;
                for (int c = 0; c < h; c++) mean += xp[(long)c * t + j] + dp[(long)c * t + j];
                mean /= h;
                double var = 0;
                for (int c = 0; c < h; c++) { double d = xp[(long)c * t + j] + dp[(long)c * t + j] - mean; var += d * d; }
                var /= h;
                float inv = (float)(1.0 / Math.Sqrt(var + 1e-5));
                for (int c = 0; c < h; c++)
                {
                    float val = xp[(long)c * t + j] + dp[(long)c * t + j];
                    op[(long)c * t + j] = ((val - (float)mean) * inv) * g[c] + be[c];
                }
            }
            return outT;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] all = [_qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _relK, _relV,
                _norm1G, _norm1B, _norm2G, _norm2B, _ffn1W, _ffn1B, _ffn2W, _ffn2B];
            foreach (Tensor? t in all) if (t is not null) yield return t;
        }
    }
}
