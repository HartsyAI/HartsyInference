using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Ssm;

/// <summary>RWKV-6 ("Finch") decoder loaded from a llama.cpp <c>rwkv6</c> GGUF. A non-transformer linear-attention
/// recurrence: each block is a <b>time-mix</b> (data-dependent token-shift via LoRA → r/k/v/g + a data-dependent
/// per-channel decay → WKV6 outer-product state recurrence → per-head GroupNorm → gate → out) and a <b>channel-mix</b>
/// (token-shift → squared-ReLU gated MLP), both with LayerNorm + residual. llama.cpp pre-divides the deep-layer
/// output weights by 2^(layer/rescale); we apply the matching runtime <c>x *= 0.5</c> every <c>rescale</c> layers.
///
/// <para>Big projections run through <see cref="IBackend.Linear"/>; the token-shift LoRAs, WKV6 scan, GroupNorm and
/// LayerNorms run host-side (the recurrence is inherently sequential).</para></summary>
public sealed unsafe class RwkvModel : IDisposable, ISsmModel
{
    private readonly GgufModelLoader.LoadedGgufModel _handle;
    private readonly IReadOnlyDictionary<string, Tensor> _w;
    private readonly List<Tensor> _keepAlive;   // originals held alive behind the no-copy relabel views
    // Per-layer recurrent state, carried across calls — see Rwkv7Model's identical pattern for the full
    // rationale. _wkvState[il]: the WKV6 outer-product state. _timeMixPrevRow/_channelMixPrevRow[il]: each
    // mix's last normed hidden row (ShiftDiff's "xx[-1]" token-shift context).
    private readonly float[][] _wkvState;
    private readonly float[][] _timeMixPrevRow;
    private readonly float[][] _channelMixPrevRow;
    private int _disposed;

    public GgufMetadata Metadata => _handle.Metadata;
    public int DModel { get; }
    public int NumLayers { get; }
    public int NumHeads { get; }
    public int HeadSize { get; }
    public int Ffn { get; }
    public int VocabSize { get; }
    public int RescaleEvery { get; }
    public float Eps { get; }

    private RwkvModel(GgufModelLoader.LoadedGgufModel handle, IReadOnlyDictionary<string, Tensor> w, List<Tensor> keepAlive,
        int dModel, int layers, int heads, int headSize, int ffn, int vocab, int rescale, float eps)
    {
        _handle = handle; _w = w; _keepAlive = keepAlive;
        DModel = dModel; NumLayers = layers; NumHeads = heads; HeadSize = headSize; Ffn = ffn;
        VocabSize = vocab; RescaleEvery = rescale; Eps = eps <= 0f ? 1e-5f : eps;
        _wkvState = new float[layers][];
        _timeMixPrevRow = new float[layers][];
        _channelMixPrevRow = new float[layers][];
        for (int i = 0; i < layers; i++)
        {
            _wkvState[i] = new float[(long)heads * headSize * headSize];
            _timeMixPrevRow[i] = new float[dModel];
            _channelMixPrevRow[i] = new float[dModel];
        }
    }

    /// <summary>Zeroes every layer's recurrent state — call before starting a new, unrelated generation.</summary>
    public void ResetState()
    {
        foreach (float[] s in _wkvState) Array.Clear(s);
        foreach (float[] r in _timeMixPrevRow) Array.Clear(r);
        foreach (float[] r in _channelMixPrevRow) Array.Clear(r);
    }

    public static RwkvModel Load(string ggufPath)
    {
        (Dictionary<string, Tensor> w, GgufModelLoader.LoadedGgufModel handle) = GgufModelLoader.LoadDequantized(ggufPath, DType.F32);
        try
        {
            // Relabel 2-D nn.Linear weights to [out, in] for backend.Linear, as NO-COPY views (Reshape shares the
            // data pointer) — a copy would double the 6.4 GB F32 model. Originals stay alive in `keep`.
            // (time_mix_w2 is 3-D → read raw; token_embd / output are read raw [vocab, d].)
            List<Tensor> keep = new();
            foreach (string key in w.Keys.ToList())
            {
                if (!key.EndsWith(".weight", StringComparison.Ordinal) || w[key].Shape.Rank != 2) continue;
                if (key.Contains("time_mix_receptance") || key.Contains("time_mix_key") || key.Contains("time_mix_value")
                    || key.Contains("time_mix_gate") || key.Contains("time_mix_output") || key.Contains("time_mix_w1")
                    || key.Contains("time_mix_decay_w1") || key.Contains("time_mix_decay_w2")
                    || key.Contains("channel_mix_key") || key.Contains("channel_mix_value") || key.Contains("channel_mix_receptance"))
                {
                    Tensor orig = w[key]; keep.Add(orig);
                    w[key] = orig.Reshape(new TensorShape((int)orig.Shape[1], (int)orig.Shape[0]));
                }
            }
            GgufMetadata m = handle.Metadata;
            int dModel = (int)m.GetUInt32("rwkv6.embedding_length");
            int layers = (int)m.GetUInt32("rwkv6.block_count");
            int headSize = (int)m.GetUInt32("rwkv6.wkv.head_size", 64u);
            int ffn = (int)m.GetUInt32("rwkv6.feed_forward_length");
            int rescale = (int)m.GetUInt32("rwkv6.rescale_every_n_layers", 0u);
            float eps = m.GetFloat32("rwkv6.attention.layer_norm_epsilon", 1e-5f);
            int vocab = (int)w["output.weight"].Shape[1];   // ne=[d, vocab], not relabeled
            return new RwkvModel(handle, w, keep, dModel, layers, dModel / headSize, headSize, ffn, vocab, rescale, eps);
        }
        catch { handle.Dispose(); throw; }
    }

    private Tensor W(string key) => _w[key];

    /// <summary>x[T,in] @ W[out,in]ᵀ → [T,out], via backend.Linear.</summary>
    private float[] Lin(IBackend backend, float[] x, int t, int inDim, string key)
    {
        Tensor wt = W(key);
        int outDim = (int)wt.Shape[0];
        using Tensor xt = new(new TensorShape(1, t, inDim), DType.F32);
        fixed (float* s = x) Buffer.MemoryCopy(s, (void*)xt.DataPointer, (long)t * inDim * 4, (long)t * inDim * 4);
        using Tensor o = new(new TensorShape(1, t, outDim), DType.F32);
        backend.Linear(o, xt, wt, null);
        backend.Sync();
        float[] r = new float[(long)t * outDim];
        fixed (float* d = r) Buffer.MemoryCopy((void*)o.DataPointer, d, r.Length * 4L, r.Length * 4L);
        return r;
    }

    private void LayerNorm(float[] x, int t, float* w, float* b)
    {
        int d = DModel;
        fixed (float* xp = x)
            for (int s = 0; s < t; s++)
            {
                float* row = xp + (long)s * d;
                double mean = 0; for (int c = 0; c < d; c++) mean += row[c]; mean /= d;
                double var = 0; for (int c = 0; c < d; c++) { double dd = row[c] - mean; var += dd * dd; } var /= d;
                float inv = (float)(1.0 / Math.Sqrt(var + Eps));
                for (int c = 0; c < d; c++) row[c] = (float)((row[c] - mean) * inv) * w[c] + b[c];
            }
    }

    /// <summary>Runs the stack over <paramref name="ids"/> — the NEW tokens since the last call — advancing each
    /// layer's carried recurrent state, and returns the next-token logits (last position). Call
    /// <see cref="ResetState"/> before the first call of a new generation.</summary>
    public float[] ForwardLastLogits(IBackend backend, IReadOnlyList<int> ids)
    {
        int seq = ids.Count, d = DModel;
        float[] x = new float[(long)seq * d];
        float* emb = (float*)W("token_embd.weight").DataPointer;
        for (int s = 0; s < seq; s++) for (int c = 0; c < d; c++) x[s * d + c] = emb[(long)ids[s] * d + c];
        LayerNorm(x, seq, (float*)W("token_embd_norm.weight").DataPointer, (float*)W("token_embd_norm.bias").DataPointer);

        for (int il = 0; il < NumLayers; il++)
        {
            TimeMix(backend, x, seq, il);
            ChannelMix(backend, x, seq, il);
            if (RescaleEvery > 0 && (il + 1) % RescaleEvery == 0)
                for (long n = 0; n < (long)seq * d; n++) x[n] *= 0.5f;
        }
        LayerNorm(x, seq, (float*)W("output_norm.weight").DataPointer, (float*)W("output_norm.bias").DataPointer);

        // lm_head host-side: logits[v] = Σ_c last[c] · output[v, c]  (output.weight is raw [vocab, d] row-major).
        float* outW = (float*)W("output.weight").DataPointer;
        float[] logits = new float[VocabSize];
        for (int v = 0; v < VocabSize; v++)
        {
            float acc = 0f; long wb = (long)v * d, xb = (long)(seq - 1) * d;
            for (int c = 0; c < d; c++) acc += x[xb + c] * outW[wb + c];
            logits[v] = acc;
        }
        return logits;
    }

    // sx[t] = xx[t-1] - xx[t]  (token-shift difference). xx[-1] is the carried last row from the previous call
    // (zero at true sequence position 0).
    private float[] ShiftDiff(float[] xx, int t, float[] prevRow)
    {
        int d = DModel; float[] sx = new float[(long)t * d];
        for (int s = 0; s < t; s++)
            for (int c = 0; c < d; c++) sx[s * d + c] = (s == 0 ? prevRow[c] : xx[(s - 1) * d + c]) - xx[s * d + c];
        Array.Copy(xx, ((long)t - 1) * d, prevRow, 0, d);
        return sx;
    }

    private void TimeMix(IBackend backend, float[] x, int t, int il)
    {
        int d = DModel, H = NumHeads, N = HeadSize, lora = (int)W($"blk.{il}.time_mix_w1.weight").Shape[0] / 5;
        string p = $"blk.{il}.";
        float[] xx = (float[])x.Clone();
        LayerNorm(xx, t, (float*)W(p + "attn_norm.weight").DataPointer, (float*)W(p + "attn_norm.bias").DataPointer);
        float[] sx = ShiftDiff(xx, t, _timeMixPrevRow[il]);

        float* lerpX = (float*)W(p + "time_mix_lerp_x.weight").DataPointer;
        float[] xmix = new float[(long)t * d];
        for (int s = 0; s < t; s++) for (int c = 0; c < d; c++) xmix[s * d + c] = xx[s * d + c] + sx[s * d + c] * lerpX[c];
        // LoRA: tanh(xmix @ w1) [t, 5*lora]; then per-component @ w2[k] → 5 × [t, d]
        float[] proj = Lin(backend, xmix, t, d, p + "time_mix_w1.weight");
        for (long n = 0; n < proj.Length; n++) proj[n] = MathF.Tanh(proj[n]);
        float* w2 = (float*)W(p + "time_mix_w2.weight").DataPointer;   // [5, d, lora] row-major
        // m[k][s,c] = sum_j proj[s, k*lora + j] * w2[k, c, j]
        float[][] m = new float[5][];
        for (int kc = 0; kc < 5; kc++)
        {
            float[] mk = new float[(long)t * d];
            for (int s = 0; s < t; s++)
                for (int c = 0; c < d; c++)
                {
                    float acc = 0f;
                    long wbase = ((long)kc * d + c) * lora;
                    long pbase = (long)s * 5 * lora + (long)kc * lora;
                    for (int j = 0; j < lora; j++) acc += proj[pbase + j] * w2[wbase + j];
                    mk[s * d + c] = acc;
                }
            m[kc] = mk;
        }
        float[] Mixed(string lerpKey, float[] mk)
        {
            float* lp = (float*)W(p + lerpKey).DataPointer;
            float[] o = new float[(long)t * d];
            for (int s = 0; s < t; s++) for (int c = 0; c < d; c++) o[s * d + c] = xx[s * d + c] + sx[s * d + c] * (lp[c] + mk[s * d + c]);
            return o;
        }
        float[] wx = Mixed("time_mix_lerp_w.weight", m[0]);
        float[] kx = Mixed("time_mix_lerp_k.weight", m[1]);
        float[] vx = Mixed("time_mix_lerp_v.weight", m[2]);
        float[] rx = Mixed("time_mix_lerp_r.weight", m[3]);
        float[] gx = Mixed("time_mix_lerp_g.weight", m[4]);
        float[] r = Lin(backend, rx, t, d, p + "time_mix_receptance.weight");
        float[] k = Lin(backend, kx, t, d, p + "time_mix_key.weight");
        float[] v = Lin(backend, vx, t, d, p + "time_mix_value.weight");
        float[] g = Lin(backend, gx, t, d, p + "time_mix_gate.weight");
        for (long n = 0; n < g.Length; n++) g[n] = g[n] / (1f + MathF.Exp(-g[n]));   // SiLU

        // decay: t_decay + (tanh(wx @ decay_w1) @ decay_w2); w = exp(-exp(.))
        float[] dlat = Lin(backend, wx, t, d, p + "time_mix_decay_w1.weight");
        for (long n = 0; n < dlat.Length; n++) dlat[n] = MathF.Tanh(dlat[n]);
        float[] wdec = Lin(backend, dlat, t, (int)W(p + "time_mix_decay_w1.weight").Shape[0], p + "time_mix_decay_w2.weight");
        float* tdecay = (float*)W(p + "time_mix_decay.weight").DataPointer;
        for (int s = 0; s < t; s++) for (int c = 0; c < d; c++) { float wv = tdecay[c] + wdec[s * d + c]; wdec[s * d + c] = MathF.Exp(-MathF.Exp(wv)); }

        // WKV6 recurrence per head: state S[H,N,N]; out[s,h,j] = Σ_i r[i]*(u[i]*k[i]*v[j] + S[i,j]); S[i,j] = w[i]*S[i,j] + k[i]*v[j]
        float* u = (float*)W(p + "time_mix_first.weight").DataPointer;   // [H,N]
        float[] state = _wkvState[il];   // carried across calls — zero only at true position 0
        float[] outv = new float[(long)t * d];
        for (int s = 0; s < t; s++)
            for (int h = 0; h < H; h++)
            {
                long hb = (long)s * d + (long)h * N, sb = (long)h * N * N, ub = (long)h * N;
                for (int j = 0; j < N; j++)
                {
                    float acc = 0f;
                    for (int i = 0; i < N; i++)
                    {
                        float kv = k[hb + i] * v[hb + j];
                        float sij = state[sb + (long)i * N + j];
                        acc += r[hb + i] * (u[ub + i] * kv + sij);
                        state[sb + (long)i * N + j] = wdec[hb + i] * sij + kv;
                    }
                    outv[hb + j] = acc;
                }
            }

        // per-head GroupNorm (eps 64e-5) * ln_w + ln_b, then * gate.
        float* lnw = (float*)W(p + "time_mix_ln.weight").DataPointer; float* lnb = (float*)W(p + "time_mix_ln.bias").DataPointer;
        for (int s = 0; s < t; s++)
            for (int h = 0; h < H; h++)
            {
                long hb = (long)s * d + (long)h * N;
                double mean = 0; for (int j = 0; j < N; j++) mean += outv[hb + j]; mean /= N;
                double var = 0; for (int j = 0; j < N; j++) { double dd = outv[hb + j] - mean; var += dd * dd; } var /= N;
                float inv = (float)(1.0 / Math.Sqrt(var + 64e-5));
                for (int j = 0; j < N; j++) { int c = h * N + j; outv[hb + j] = ((float)((outv[hb + j] - mean) * inv) * lnw[c] + lnb[c]) * g[hb + j]; }
            }
        float[] o2 = Lin(backend, outv, t, d, p + "time_mix_output.weight");
        for (long n = 0; n < (long)t * d; n++) x[n] += o2[n];
    }

    private void ChannelMix(IBackend backend, float[] x, int t, int il)
    {
        int d = DModel;
        string p = $"blk.{il}.";
        float[] xx = (float[])x.Clone();
        LayerNorm(xx, t, (float*)W(p + "attn_norm_2.weight").DataPointer, (float*)W(p + "attn_norm_2.bias").DataPointer);
        float[] sx = ShiftDiff(xx, t, _channelMixPrevRow[il]);
        float* lk = (float*)W(p + "channel_mix_lerp_k.weight").DataPointer; float* lr = (float*)W(p + "channel_mix_lerp_r.weight").DataPointer;
        float[] kx = new float[(long)t * d], rx = new float[(long)t * d];
        for (int s = 0; s < t; s++) for (int c = 0; c < d; c++) { kx[s * d + c] = xx[s * d + c] + sx[s * d + c] * lk[c]; rx[s * d + c] = xx[s * d + c] + sx[s * d + c] * lr[c]; }
        float[] kk = Lin(backend, kx, t, d, p + "channel_mix_key.weight");   // [t, ffn]
        for (long n = 0; n < kk.Length; n++) { float relu = MathF.Max(kk[n], 0f); kk[n] = relu * relu; }   // squared ReLU
        float[] kv = Lin(backend, kk, t, Ffn, p + "channel_mix_value.weight");   // [t, d]
        float[] rr = Lin(backend, rx, t, d, p + "channel_mix_receptance.weight");
        for (long n = 0; n < (long)t * d; n++) x[n] += (1f / (1f + MathF.Exp(-rr[n]))) * kv[n];
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _handle.Dispose();
    }
}
