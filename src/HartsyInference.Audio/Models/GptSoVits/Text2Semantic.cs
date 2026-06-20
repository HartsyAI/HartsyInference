using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Dia;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.GptSoVits;

/// <summary>GPT-SoVITS T2S decoder: assembles <c>[text+bert ‖ reference-semantic-prefix]</c>, runs a
/// post-LayerNorm transformer (sinusoidal positions, ReLU MLP, biased fused-QKV attention), and
/// autoregressively predicts target semantic tokens until EOS. Reuses <see cref="DiaHeads"/> for head
/// reshapes, <see cref="WhisperOps.ProjectLinear"/>, and <see cref="NucleusSampler"/> (top-k + repetition
/// penalty). Full-sequence re-feed (no KV cache) — correct, perf-tunable later.</summary>
public sealed unsafe class Text2Semantic : IDisposable
{
    private readonly Text2SemanticConfig _cfg;
    private Tensor? _textEmb, _audioEmb, _bertW, _bertB, _predW;
    private float _alphaText = 1f, _alphaAudio = 1f;
    private readonly Block[] _layers;
    private int _disposed;

    public Text2Semantic(Text2SemanticConfig cfg)
    {
        _cfg = cfg;
        _layers = new Block[cfg.NumLayers];
        for (int i = 0; i < _layers.Length; i++) _layers[i] = new Block(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "model")
    {
        _textEmb = WhisperOps.EnsureF32(w[$"{prefix}.ar_text_embedding.word_embeddings.weight"]);
        _audioEmb = WhisperOps.EnsureF32(w[$"{prefix}.ar_audio_embedding.word_embeddings.weight"]);
        _bertW = WhisperOps.EnsureF32(w[$"{prefix}.bert_proj.weight"]); _bertB = WhisperOps.EnsureF32(w[$"{prefix}.bert_proj.bias"]);
        _predW = WhisperOps.EnsureF32(w[$"{prefix}.ar_predict_layer.weight"]);
        _alphaText = ((float*)WhisperOps.EnsureF32(w[$"{prefix}.ar_text_position.alpha"]).DataPointer)[0];
        _alphaAudio = ((float*)WhisperOps.EnsureF32(w[$"{prefix}.ar_audio_position.alpha"]).DataPointer)[0];
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, $"{prefix}.h.layers.{i}");
    }

    /// <summary>Autoregressively generates target semantic tokens. <paramref name="phonemes"/> + the BERT
    /// feature <c>[BertDim, Tx]</c> form the (bidirectional) text span; <paramref name="prefix"/> is the
    /// reference's semantic tokens. Returns the generated tokens (EOS stripped).</summary>
    public List<int> Generate(IBackend backend, ReadOnlySpan<int> phonemes, Tensor bert, ReadOnlySpan<int> prefix,
        int maxNew, int seed = 0)
    {
        int h = _cfg.Hidden, tx = phonemes.Length;
        // Text span: text_emb + bert_proj, then sinusoidal positions.
        Tensor xText = BuildTextEmbed(backend, phonemes, bert, tx);

        List<int> y = new(prefix.Length + maxNew);
        for (int i = 0; i < prefix.Length; i++) y.Add(prefix[i]);
        uint rng = DeterministicRng.Seed(seed);
        HashSet<int> seen = new();

        for (int step = 0; step < maxNew; step++)
        {
            int ty = y.Count;
            Tensor seq = AssembleSequence(xText, tx, y, ty, h);
            int total = tx + ty;
            Tensor mask = BuildMask(tx, ty);
            Tensor hid = seq;
            for (int i = 0; i < _layers.Length; i++) { Tensor n = _layers[i].Forward(backend, hid, total, mask); hid.Dispose(); hid = n; }
            mask.Dispose();

            Tensor last = SliceLast(hid, h); hid.Dispose();
            Tensor logitsT = WhisperOps.ProjectLinear(backend, last, _predW!, null, 1, 1, h, _cfg.SemanticVocab);
            last.Dispose();
            Span<float> logits = new((void*)logitsT.DataPointer, _cfg.SemanticVocab);
            if (_cfg.RepetitionPenalty != 1f)
                foreach (int tok in seen) logits[tok] = logits[tok] > 0 ? logits[tok] / _cfg.RepetitionPenalty : logits[tok] * _cfg.RepetitionPenalty;
            int next = NucleusSampler.Draw(logits, _cfg.SemanticVocab, _cfg.Temperature, _cfg.TopK, 0f, ref rng);
            logitsT.Dispose();
            if (next == _cfg.EosToken) break;
            y.Add(next); seen.Add(next);
        }
        xText.Dispose();
        return y.GetRange(prefix.Length, y.Count - prefix.Length);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_textEmb, _audioEmb, _bertW, _bertB, _predW];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        foreach (Block b in _layers) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    private Tensor BuildTextEmbed(IBackend backend, ReadOnlySpan<int> phonemes, Tensor bert, int tx)
    {
        int h = _cfg.Hidden;
        Tensor x = new(new TensorShape(1, tx, h), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* ep = (float*)_textEmb!.DataPointer;
        for (int j = 0; j < tx; j++)
            for (int c = 0; c < h; c++) xp[(long)j * h + c] = ep[(long)phonemes[j] * h + c];
        // bert_proj(bert [BertDim, Tx]) → [Tx, h], added.
        Tensor bertCl = new(new TensorShape(1, tx, _cfg.BertDim), DType.F32);
        backend.Transpose2D(bertCl, bert, _cfg.BertDim, tx);
        Tensor bproj = WhisperOps.ProjectLinear(backend, bertCl, _bertW!, _bertB, 1, tx, _cfg.BertDim, h);
        bertCl.Dispose();
        float* bp = (float*)bproj.DataPointer;
        for (long n = 0; n < (long)tx * h; n++) xp[n] += bp[n];
        bproj.Dispose();
        AddSinusoidal(xp, tx, h, _alphaText);
        return x;
    }

    private Tensor AssembleSequence(Tensor xText, int tx, List<int> y, int ty, int h)
    {
        Tensor seq = new(new TensorShape(1, tx + ty, h), DType.F32);
        float* sp = (float*)seq.DataPointer;
        Buffer.MemoryCopy((void*)xText.DataPointer, sp, (long)tx * h * 4, (long)tx * h * 4);
        // Audio prefix embeds + sinusoidal positions (positions start at 0 for the audio span).
        float* ap = (float*)_audioEmb!.DataPointer;
        float* yp = sp + (long)tx * h;
        for (int j = 0; j < ty; j++)
            for (int c = 0; c < h; c++) yp[(long)j * h + c] = ap[(long)y[j] * h + c];
        AddSinusoidal(yp, ty, h, _alphaAudio);
        return seq;
    }

    private static void AddSinusoidal(float* x, int t, int h, float alpha)
    {
        float scale = MathF.Sqrt(h);
        for (int j = 0; j < t; j++)
            for (int c = 0; c < h; c++)
            {
                int pair = c / 2;
                float div = MathF.Exp(-(2f * pair / h) * MathF.Log(10000f));
                float pe = (c % 2 == 0) ? MathF.Sin(j * div) : MathF.Cos(j * div);
                x[(long)j * h + c] = x[(long)j * h + c] * scale + alpha * pe;
            }
    }

    private static Tensor BuildMask(int tx, int ty)
    {
        int total = tx + ty;
        Tensor mask = new(new TensorShape(1, 1, total, total), DType.F32);
        float* p = (float*)mask.DataPointer;
        const float NegInf = -1e30f;
        for (int q = 0; q < total; q++)
            for (int k = 0; k < total; k++)
            {
                // Text span attends bidirectionally within text; audio span is causal over the whole sequence.
                bool allowed = q < tx ? k < tx : k <= q;
                p[q * total + k] = allowed ? 0f : NegInf;
            }
        return mask;
    }

    private static Tensor SliceLast(Tensor hidden, int h)
    {
        int t = (int)hidden.Shape[1];
        Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        Buffer.MemoryCopy((float*)hidden.DataPointer + (long)(t - 1) * h, (void*)last.DataPointer, h * 4, h * 4);
        return last;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    /// <summary>One post-LayerNorm block: <c>x = LN1(x + attn(x)); x = LN2(x + relu_ffn(x))</c>.</summary>
    private sealed class Block
    {
        private readonly Text2SemanticConfig _cfg;
        private Tensor? _inW, _inB, _outW, _outB, _l1W, _l1B, _l2W, _l2B, _n1W, _n1B, _n2W, _n2B;

        public Block(Text2SemanticConfig cfg) => _cfg = cfg;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _inW = WhisperOps.EnsureF32(w[$"{p}.self_attn.in_proj_weight"]); _inB = WhisperOps.EnsureF32(w[$"{p}.self_attn.in_proj_bias"]);
            _outW = WhisperOps.EnsureF32(w[$"{p}.self_attn.out_proj.weight"]); _outB = WhisperOps.EnsureF32(w[$"{p}.self_attn.out_proj.bias"]);
            _l1W = WhisperOps.EnsureF32(w[$"{p}.linear1.weight"]); _l1B = WhisperOps.EnsureF32(w[$"{p}.linear1.bias"]);
            _l2W = WhisperOps.EnsureF32(w[$"{p}.linear2.weight"]); _l2B = WhisperOps.EnsureF32(w[$"{p}.linear2.bias"]);
            _n1W = WhisperOps.EnsureF32(w[$"{p}.norm1.weight"]); _n1B = WhisperOps.EnsureF32(w[$"{p}.norm1.bias"]);
            _n2W = WhisperOps.EnsureF32(w[$"{p}.norm2.weight"]); _n2B = WhisperOps.EnsureF32(w[$"{p}.norm2.bias"]);
        }

        public Tensor Forward(IBackend backend, Tensor x, int t, Tensor mask)
        {
            int h = _cfg.Hidden, heads = _cfg.NumHeads, hd = _cfg.HeadDim;
            // Fused in_proj [3h, h] → q/k/v.
            Tensor qkv = WhisperOps.ProjectLinear(backend, x, _inW!, _inB, 1, t, h, 3 * h);
            Tensor qM = new(new TensorShape(1, heads, t, hd), DType.F32);
            Tensor kM = new(new TensorShape(1, heads, t, hd), DType.F32);
            Tensor vM = new(new TensorShape(1, heads, t, hd), DType.F32);
            float* p = (float*)qkv.DataPointer;
            Split(p, qM, 0, t, heads, hd, h); Split(p, kM, h, t, heads, hd, h); Split(p, vM, 2 * h, t, heads, hd, h);
            qkv.Dispose();
            Tensor attn = new(new TensorShape(1, heads, t, hd), DType.F32);
            backend.ScaledDotProductAttention(attn, qM, kM, vM, mask, 1f / MathF.Sqrt(hd));
            qM.Dispose(); kM.Dispose(); vM.Dispose();
            Tensor flat = new(new TensorShape(1, t, h), DType.F32);
            DiaHeads.HeadsToFlat(flat, attn, t, heads, hd); attn.Dispose();
            Tensor attnOut = WhisperOps.ProjectLinear(backend, flat, _outW!, _outB, 1, t, h, h); flat.Dispose();

            Tensor afterAttn = new(x.Shape, DType.F32);
            backend.Add(afterAttn, x, attnOut); attnOut.Dispose();
            Tensor n1 = new(x.Shape, DType.F32);
            backend.LayerNorm(n1, afterAttn, _n1W!, _n1B!, _cfg.NormEps); afterAttn.Dispose();

            Tensor f1 = WhisperOps.ProjectLinear(backend, n1, _l1W!, _l1B, 1, t, h, _cfg.FfnDim);
            float* fp = (float*)f1.DataPointer;
            for (long n = 0; n < f1.ElementCount; n++) if (fp[n] < 0) fp[n] = 0;   // ReLU
            Tensor f2 = WhisperOps.ProjectLinear(backend, f1, _l2W!, _l2B, 1, t, _cfg.FfnDim, h); f1.Dispose();
            Tensor afterFf = new(x.Shape, DType.F32);
            backend.Add(afterFf, n1, f2); n1.Dispose(); f2.Dispose();
            Tensor outT = new(x.Shape, DType.F32);
            backend.LayerNorm(outT, afterFf, _n2W!, _n2B!, _cfg.NormEps); afterFf.Dispose();
            return outT;
        }

        private static void Split(float* qkv, Tensor dst, int colOff, int t, int heads, int hd, int h)
        {
            float* dp = (float*)dst.DataPointer;
            for (int s = 0; s < t; s++)
                for (int hh = 0; hh < heads; hh++)
                {
                    long src = (long)s * 3 * h + colOff + (long)hh * hd;
                    long d = ((long)hh * t + s) * hd;
                    Buffer.MemoryCopy(qkv + src, dp + d, hd * 4, hd * 4);
                }
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] all = [_inW, _inB, _outW, _outB, _l1W, _l1B, _l2W, _l2B, _n1W, _n1B, _n2W, _n2B];
            foreach (Tensor? t in all) if (t is not null) yield return t;
        }
    }
}
