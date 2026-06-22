using HartsyInference.Audio.Models.Vits;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.GptSoVits;

/// <summary>GPT-SoVITS SoVITS prior encoder (the real <c>enc_p</c>). Maps the dequantized HuBERT semantic
/// latent <c>[1, 768, T]</c> to the prior <c>(m_p, logs_p)</c> the flow consumes:
/// <list type="number">
/// <item><c>ssl_proj</c> Conv1d 768→192 over the latent.</item>
/// <item><c>encoder_ssl</c> — 3 VITS relative-position attention layers.</item>
/// <item><c>mrte</c> — Multi-Reference Timbre Encoder: cross-attention with the SSL stream as query and the
/// text-embedding stream as key/value, channels 192→512→192, 4 heads, fused with the speaker embedding
/// <c>ge</c>. At inference the "text" key/value is the SoVITS text embedding of the semantic tokens passed
/// through <c>encoder_text</c>.</item>
/// <item><c>encoder2</c> — 3 more VITS attention layers.</item>
/// <item><c>proj</c> Conv1d 192→384, split into <c>m_p / logs_p</c>.</item>
/// </list>
///
/// <para><b>Reuse:</b> the VITS rel-pos attention layer is replicated from <see cref="VitsTextEncoder"/>
/// (its private <c>Layer</c> can't be shared without editing it); convs/attention via
/// <see cref="IBackend"/>. The MRTE cross-attention block is built explicitly here.</para></summary>
public sealed unsafe class SoVitsEncP
{
    private readonly int _hidden, _inter, _mrteHidden, _mrteHeads;
    private readonly VitsConfig _cfg;
    private readonly AttnLayer[] _encoderSsl;
    private readonly AttnLayer[] _encoderText;
    private readonly AttnLayer[] _encoder2;
    private Tensor? _sslProjW, _sslProjB, _projW, _projB, _textEmb;
    private Tensor? _mrteCrossQW, _mrteCrossQB, _mrteCrossKW, _mrteCrossKB, _mrteCrossVW, _mrteCrossVB;
    private Tensor? _mrteCrossOW, _mrteCrossOB, _mrteCInW, _mrteCInB, _mrteCOutW, _mrteCOutB, _mrteGeW, _mrteGeB;

    public SoVitsEncP(VitsConfig cfg, int sslDim = 768, int sslLayers = 3, int textLayers = 6,
        int enc2Layers = 3, int mrteHidden = 512, int mrteHeads = 4)
    {
        _cfg = cfg; _hidden = cfg.HiddenChannels; _inter = cfg.InterChannels;
        _mrteHidden = mrteHidden; _mrteHeads = mrteHeads;
        _ = sslDim;     // ssl input width is taken from the ssl_proj weight shape; param kept for the call contract
        int kc = _hidden / cfg.NumHeads;
        _encoderSsl = Build(sslLayers, kc);
        _encoderText = Build(textLayers, kc);
        _encoder2 = Build(enc2Layers, kc);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "enc_p")
    {
        _sslProjW = VitsWeights.Conv(w, $"{prefix}.ssl_proj"); _sslProjB = VitsWeights.Bias(w, $"{prefix}.ssl_proj");
        _textEmb = WhisperOps.EnsureF32(w[$"{prefix}.text_embedding.weight"]);
        for (int i = 0; i < _encoderSsl.Length; i++) _encoderSsl[i].LoadWeights(w, $"{prefix}.encoder_ssl", i);
        for (int i = 0; i < _encoderText.Length; i++) _encoderText[i].LoadWeights(w, $"{prefix}.encoder_text", i);
        for (int i = 0; i < _encoder2.Length; i++) _encoder2[i].LoadWeights(w, $"{prefix}.encoder2", i);
        _projW = VitsWeights.Conv(w, $"{prefix}.proj"); _projB = VitsWeights.Bias(w, $"{prefix}.proj");

        string m = $"{prefix}.mrte";
        _mrteCInW = VitsWeights.Conv(w, $"{m}.c_pre"); _mrteCInB = VitsWeights.Bias(w, $"{m}.c_pre");
        _mrteCOutW = VitsWeights.Conv(w, $"{m}.c_post"); _mrteCOutB = VitsWeights.Bias(w, $"{m}.c_post");
        _mrteCrossQW = VitsWeights.Conv(w, $"{m}.cross_attention.conv_q"); _mrteCrossQB = VitsWeights.Bias(w, $"{m}.cross_attention.conv_q");
        _mrteCrossKW = VitsWeights.Conv(w, $"{m}.cross_attention.conv_k"); _mrteCrossKB = VitsWeights.Bias(w, $"{m}.cross_attention.conv_k");
        _mrteCrossVW = VitsWeights.Conv(w, $"{m}.cross_attention.conv_v"); _mrteCrossVB = VitsWeights.Bias(w, $"{m}.cross_attention.conv_v");
        _mrteCrossOW = VitsWeights.Conv(w, $"{m}.cross_attention.conv_o"); _mrteCrossOB = VitsWeights.Bias(w, $"{m}.cross_attention.conv_o");
        if (w.ContainsKey($"{m}.text_pre.weight"))
        {
            _mrteGeW = VitsWeights.Conv(w, $"{m}.text_pre"); _mrteGeB = VitsWeights.Bias(w, $"{m}.text_pre");
        }
        else if (w.ContainsKey($"{m}.ge_proj.weight"))
        {
            _mrteGeW = VitsWeights.Conv(w, $"{m}.ge_proj"); _mrteGeB = VitsWeights.Bias(w, $"{m}.ge_proj");
        }
    }

    /// <summary>Runs the prior encoder. <paramref name="sslLatent"/> is the dequantized HuBERT latent
    /// <c>[1, sslDim, t]</c>; <paramref name="textTokens"/> are the semantic ids embedded via
    /// <c>text_embedding</c> and refined by <c>encoder_text</c> to form the MRTE key/value stream;
    /// <paramref name="ge"/> is the speaker embedding <c>[1, gin, 1]</c>.</summary>
    public (Tensor MP, Tensor LogsP) Forward(IBackend backend, Tensor sslLatent, int t,
        ReadOnlySpan<int> textTokens, Tensor ge)
    {
        Tensor x = new(new TensorShape(1, _hidden, t), DType.F32);
        backend.Conv1d(x, sslLatent, _sslProjW!, _sslProjB, 1, 0, 0, 1, 1);
        for (int i = 0; i < _encoderSsl.Length; i++)
        {
            Tensor next = _encoderSsl[i].Forward(backend, x, t); x.Dispose(); x = next;
        }

        // Text stream: embed semantic tokens [1, hidden, tt] → encoder_text.
        int tt = textTokens.Length;
        Tensor text = EmbedText(textTokens, tt);
        for (int i = 0; i < _encoderText.Length; i++)
        {
            Tensor next = _encoderText[i].Forward(backend, text, tt); text.Dispose(); text = next;
        }

        Tensor fused = Mrte(backend, x, t, text, tt, ge); x.Dispose(); text.Dispose();
        for (int i = 0; i < _encoder2.Length; i++)
        {
            Tensor next = _encoder2[i].Forward(backend, fused, t); fused.Dispose(); fused = next;
        }

        Tensor stats = new(new TensorShape(1, 2 * _inter, t), DType.F32);
        backend.Conv1d(stats, fused, _projW!, _projB, 1, 0, 0, 1, 1); fused.Dispose();
        Tensor mP = new(new TensorShape(1, _inter, t), DType.F32);
        Tensor logsP = new(new TensorShape(1, _inter, t), DType.F32);
        float* sp = (float*)stats.DataPointer;
        Buffer.MemoryCopy(sp, (void*)mP.DataPointer, (long)_inter * t * 4, (long)_inter * t * 4);
        Buffer.MemoryCopy(sp + (long)_inter * t, (void*)logsP.DataPointer, (long)_inter * t * 4, (long)_inter * t * 4);
        stats.Dispose();
        return (mP, logsP);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own =
        [
            _sslProjW, _sslProjB, _projW, _projB, _textEmb,
            _mrteCrossQW, _mrteCrossQB, _mrteCrossKW, _mrteCrossKB, _mrteCrossVW, _mrteCrossVB,
            _mrteCrossOW, _mrteCrossOB, _mrteCInW, _mrteCInB, _mrteCOutW, _mrteCOutB, _mrteGeW, _mrteGeB,
        ];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        foreach (AttnLayer l in _encoderSsl) foreach (Tensor t in l.EnumerateWeights()) yield return t;
        foreach (AttnLayer l in _encoderText) foreach (Tensor t in l.EnumerateWeights()) yield return t;
        foreach (AttnLayer l in _encoder2) foreach (Tensor t in l.EnumerateWeights()) yield return t;
    }

    private Tensor EmbedText(ReadOnlySpan<int> tokens, int tt)
    {
        int dim = (int)_textEmb!.Shape[1];
        float scale = MathF.Sqrt(dim);
        Tensor x = new(new TensorShape(1, _hidden, tt), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* ep = (float*)_textEmb.DataPointer;
        for (int j = 0; j < tt; j++)
            for (int c = 0; c < _hidden; c++)
                xp[(long)c * tt + j] = ep[(long)tokens[j] * dim + c] * scale;
        return x;
    }

    /// <summary>MRTE: project SSL query + text key/value into <c>mrte_hidden</c>, cross-attend (SSL queries,
    /// text key/value), project back to <c>hidden</c>, then add the broadcast speaker embedding <c>ge</c>.</summary>
    private Tensor Mrte(IBackend backend, Tensor ssl, int t, Tensor text, int tt, Tensor ge)
    {
        // Lift SSL into the MRTE channel space (c_pre: hidden → mrte_hidden).
        Tensor q = new(new TensorShape(1, _mrteHidden, t), DType.F32);
        backend.Conv1d(q, ssl, _mrteCrossQW!, _mrteCrossQB, 1, 0, 0, 1, 1);
        Tensor k = new(new TensorShape(1, _mrteHidden, tt), DType.F32);
        backend.Conv1d(k, text, _mrteCrossKW!, _mrteCrossKB, 1, 0, 0, 1, 1);
        Tensor v = new(new TensorShape(1, _mrteHidden, tt), DType.F32);
        backend.Conv1d(v, text, _mrteCrossVW!, _mrteCrossVB, 1, 0, 0, 1, 1);

        Tensor attn = CrossAttention(q, t, k, v, tt);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor o = new(new TensorShape(1, _mrteHidden, t), DType.F32);
        backend.Conv1d(o, attn, _mrteCrossOW!, _mrteCrossOB, 1, 0, 0, 1, 1); attn.Dispose();

        // c_post: mrte_hidden → hidden, then add the (channel-projected) speaker embedding broadcast over time.
        Tensor outT = new(new TensorShape(1, _hidden, t), DType.F32);
        backend.Conv1d(outT, o, _mrteCOutW!, _mrteCOutB, 1, 0, 0, 1, 1); o.Dispose();
        if (_mrteGeW is not null)
        {
            Tensor geProj = new(new TensorShape(1, _hidden, 1), DType.F32);
            backend.Conv1d(geProj, ge, _mrteGeW, _mrteGeB, 1, 0, 0, 1, 1);
            float* op = (float*)outT.DataPointer;
            float* gp = (float*)geProj.DataPointer;
            for (int c = 0; c < _hidden; c++)
                for (int j = 0; j < t; j++) op[(long)c * t + j] += gp[c];
            geProj.Dispose();
        }
        return outT;
    }

    /// <summary>Multi-head cross-attention: query <c>[1, mrte_hidden, t]</c> attends over key/value
    /// <c>[1, mrte_hidden, tt]</c>. Channels-first throughout; output <c>[1, mrte_hidden, t]</c>.</summary>
    private Tensor CrossAttention(Tensor q, int t, Tensor k, Tensor v, int tt)
    {
        int heads = _mrteHeads, kc = _mrteHidden / heads;
        float invSqrt = 1f / MathF.Sqrt(kc);
        Tensor outT = new(new TensorShape(1, _mrteHidden, t), DType.F32);
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
        float* op = (float*)outT.DataPointer;
        float[] scores = new float[tt];
        for (int hd = 0; hd < heads; hd++)
        {
            int chBase = hd * kc;
            for (int i = 0; i < t; i++)
            {
                float maxS = float.NegativeInfinity;
                for (int j = 0; j < tt; j++)
                {
                    float dot = 0f;
                    for (int c = 0; c < kc; c++)
                        dot += qp[(long)(chBase + c) * t + i] * kp[(long)(chBase + c) * tt + j];
                    float s = dot * invSqrt;
                    scores[j] = s;
                    if (s > maxS) maxS = s;
                }
                float sum = 0f;
                for (int j = 0; j < tt; j++) { scores[j] = MathF.Exp(scores[j] - maxS); sum += scores[j]; }
                float inv = 1f / sum;
                for (int c = 0; c < kc; c++)
                {
                    float acc = 0f;
                    for (int j = 0; j < tt; j++) acc += scores[j] * inv * vp[(long)(chBase + c) * tt + j];
                    op[(long)(chBase + c) * t + i] = acc;
                }
            }
        }
        return outT;
    }

    private AttnLayer[] Build(int n, int kc)
    {
        AttnLayer[] layers = new AttnLayer[n];
        for (int i = 0; i < n; i++) layers[i] = new AttnLayer(_cfg, kc);
        return layers;
    }

    /// <summary>VITS relative-position attention encoder layer (replicated from <see cref="VitsTextEncoder"/>'s
    /// private <c>Layer</c>; identical math, kept self-contained so enc_p doesn't reach into that type).</summary>
    private sealed class AttnLayer
    {
        private readonly VitsConfig _cfg;
        private readonly int _kCh, _w;
        private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _relK, _relV;
        private Tensor? _norm1G, _norm1B, _norm2G, _norm2B, _ffn1W, _ffn1B, _ffn2W, _ffn2B;

        public AttnLayer(VitsConfig cfg, int kCh) { _cfg = cfg; _kCh = kCh; _w = cfg.WindowSize; }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, int i)
        {
            _qW = VitsWeights.Conv(w, $"{prefix}.attn_layers.{i}.conv_q"); _qB = VitsWeights.Bias(w, $"{prefix}.attn_layers.{i}.conv_q");
            _kW = VitsWeights.Conv(w, $"{prefix}.attn_layers.{i}.conv_k"); _kB = VitsWeights.Bias(w, $"{prefix}.attn_layers.{i}.conv_k");
            _vW = VitsWeights.Conv(w, $"{prefix}.attn_layers.{i}.conv_v"); _vB = VitsWeights.Bias(w, $"{prefix}.attn_layers.{i}.conv_v");
            _oW = VitsWeights.Conv(w, $"{prefix}.attn_layers.{i}.conv_o"); _oB = VitsWeights.Bias(w, $"{prefix}.attn_layers.{i}.conv_o");
            _relK = w[$"{prefix}.attn_layers.{i}.emb_rel_k"]; _relV = w[$"{prefix}.attn_layers.{i}.emb_rel_v"];
            _norm1G = w[$"{prefix}.norm_layers_1.{i}.gamma"]; _norm1B = w[$"{prefix}.norm_layers_1.{i}.beta"];
            _norm2G = w[$"{prefix}.norm_layers_2.{i}.gamma"]; _norm2B = w[$"{prefix}.norm_layers_2.{i}.beta"];
            _ffn1W = VitsWeights.Conv(w, $"{prefix}.ffn_layers.{i}.conv_1"); _ffn1B = VitsWeights.Bias(w, $"{prefix}.ffn_layers.{i}.conv_1");
            _ffn2W = VitsWeights.Conv(w, $"{prefix}.ffn_layers.{i}.conv_2"); _ffn2B = VitsWeights.Bias(w, $"{prefix}.ffn_layers.{i}.conv_2");
        }

        public Tensor Forward(IBackend backend, Tensor x, int t)
        {
            int h = _cfg.HiddenChannels;
            Tensor attn = Attention(backend, x, t);
            Tensor afterAttn = AddNorm(x, attn, _norm1G!, _norm1B!, h, t); attn.Dispose();
            Tensor f1 = new(new TensorShape(1, _cfg.FilterChannels, t), DType.F32);
            backend.Conv1d(f1, afterAttn, _ffn1W!, _ffn1B, 1, 1, 1, 1, 1);
            float* fp = (float*)f1.DataPointer;
            for (long n = 0; n < f1.ElementCount; n++) if (fp[n] < 0) fp[n] = 0;
            Tensor f2 = new(new TensorShape(1, h, t), DType.F32);
            backend.Conv1d(f2, f1, _ffn2W!, _ffn2B, 1, 1, 1, 1, 1); f1.Dispose();
            Tensor outT = AddNorm(afterAttn, f2, _norm2G!, _norm2B!, h, t);
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
            float* rk = (float*)_relK!.DataPointer, rv = (float*)_relV!.DataPointer;
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
                        int rel = Math.Clamp(j - i, -w, w) + w;
                        float dot = 0, relDot = 0;
                        for (int c = 0; c < kc; c++)
                        {
                            float qv = qp[(long)(chBase + c) * t + i];
                            dot += qv * kp[(long)(chBase + c) * t + j];
                            relDot += qv * rk[(long)rel * kc + c];
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
                            int rel = Math.Clamp(j - i, -w, w) + w;
                            acc += p * (vp[(long)(chBase + c) * t + j] + rv[(long)rel * kc + c]);
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

        private static Tensor AddNorm(Tensor x, Tensor delta, Tensor gamma, Tensor beta, int h, int t)
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
