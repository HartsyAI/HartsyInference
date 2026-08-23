using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Ssm;

/// <summary>RWKV-7 ("Goose") decoder loaded from a llama.cpp <c>rwkv7</c> GGUF: a non-transformer recurrence built on the generalized <b>delta rule</b> (vs RWKV-6's WKV6 outer-product).</summary>
/// <remarks>Each block is a <b>time-mix</b> (fused token-shift lerp for r/w/k/v/a/g → receptance/key/value + a data-dependent decay <c>w</c>, an in-context-learning-rate <c>a</c>, a value-residual mix, and a gate, all via small LoRAs → an L2-normalized key <c>kk</c> and modified key <c>k</c> → the <b>WKV7 delta-rule state recurrence</b> <c>S = S·w + v⊗k + (a·S)⊗b</c> with <c>a=−kk, b=kk·iclr</c>, out = S·r → per-head GroupNorm → r·k bonus → gate → out_proj) and a <b>channel-mix</b> (token-shift → squared-ReLU MLP, no receptance), LayerNorm + residual around each. Big projections run through <see cref="IBackend.Linear"/>; the LoRAs, scan, GroupNorm and LayerNorms run host-side (the recurrence is sequential).</remarks>
public sealed unsafe class Rwkv7Model : IDisposable, ISsmModel
{
    private readonly GgufModelLoader.LoadedGgufModel _handle;
    private readonly IReadOnlyDictionary<string, Tensor> _w;
    private readonly List<Tensor> _keepAlive;
    // Per-layer recurrent state, carried across calls so ForwardLastLogits can be fed only the NEW tokens each
    // call (true O(1)-per-token decode) instead of recomputing the whole growing sequence every step.
    // _wkvState[il] is the WKV7 delta-rule state S; _timeMixPrevRow/_channelMixPrevRow[il] are each mix's last
    // normed hidden row from the previous call (ShiftDiff's "xx[-1]" token-shift context). All start zeroed
    // (matches the original zero-shift / zero-initial-state at true sequence position 0); ResetState() re-zeros
    // them for a new generation (the model instance is reused across unrelated chat turns via the device slot).
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
    public float Eps { get; }

    private Rwkv7Model(GgufModelLoader.LoadedGgufModel handle, IReadOnlyDictionary<string, Tensor> w, List<Tensor> keepAlive,
        int dModel, int layers, int heads, int headSize, int ffn, int vocab, float eps)
    {
        _handle = handle; _w = w; _keepAlive = keepAlive;
        DModel = dModel; NumLayers = layers; NumHeads = heads; HeadSize = headSize; Ffn = ffn;
        VocabSize = vocab; Eps = eps <= 0f ? 1e-5f : eps;
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

    /// <summary>Zeroes every layer's recurrent state — call before starting a new, unrelated generation, since the model instance persists across chat turns via the provider's device slot.</summary>
    public void ResetState()
    {
        foreach (float[] s in _wkvState) Array.Clear(s);
        foreach (float[] r in _timeMixPrevRow) Array.Clear(r);
        foreach (float[] r in _channelMixPrevRow) Array.Clear(r);
    }

    public static Rwkv7Model Load(string ggufPath)
    {
        (Dictionary<string, Tensor> w, GgufModelLoader.LoadedGgufModel handle) = GgufModelLoader.LoadDequantized(ggufPath, DType.F32);
        try
        {
            // Relabel the 2-D nn.Linear weights to [out, in] for backend.Linear, as NO-COPY views (the originals
            // stay alive in `keep`). The LoRA matrices (w1/w2/a1/a2/v1/v2/g1/g2) and the main/channel projections.
            // 1-D vectors (w0/a0/v0/k_k/k_a/r_k/lerp/ln) + token_embd/output are read raw.
            string[] relabel = ["time_mix_receptance", "time_mix_key", "time_mix_value", "time_mix_output",
                "time_mix_w1", "time_mix_w2", "time_mix_a1", "time_mix_a2", "time_mix_v1", "time_mix_v2",
                "time_mix_g1", "time_mix_g2", "channel_mix_key", "channel_mix_value"];
            List<Tensor> keep = new();
            foreach (string key in w.Keys.ToList())
            {
                if (!key.EndsWith(".weight", StringComparison.Ordinal) || w[key].Shape.Rank != 2) continue;
                if (relabel.Any(r => key.Contains(r)))
                {
                    Tensor orig = w[key]; keep.Add(orig);
                    w[key] = orig.Reshape(new TensorShape((int)orig.Shape[1], (int)orig.Shape[0]));
                }
            }
            GgufMetadata m = handle.Metadata;
            int dModel = (int)m.GetUInt32("rwkv7.embedding_length");
            int layers = (int)m.GetUInt32("rwkv7.block_count");
            int headSize = (int)m.GetUInt32("rwkv7.wkv.head_size", 64u);
            int ffn = (int)m.GetUInt32("rwkv7.feed_forward_length");
            float eps = m.GetFloat32("rwkv7.attention.layer_norm_epsilon", 1e-5f);
            int vocab = (int)w["output.weight"].Shape[1];   // ne=[d, vocab], raw
            return new Rwkv7Model(handle, w, keep, dModel, layers, dModel / headSize, headSize, ffn, vocab, eps);
        }
        catch { handle.Dispose(); throw; }
    }

    private Tensor W(string key) => _w[key];

    private float[] Lin(IBackend backend, float[] x, int t, int inDim, string key) => RwkvOps.Lin(backend, x, t, inDim, W(key));

    private void LayerNorm(float[] x, int t, float* w, float* b) => RwkvOps.LayerNorm(x, t, DModel, Eps, w, b);

    /// <summary>Runs the stack over <paramref name="ids"/> — the NEW tokens since the last call — advancing each layer's carried recurrent state, and returns the next-token logits (last position); call <see cref="ResetState"/> before the first call of a new generation.</summary>
    public float[] ForwardLastLogits(IBackend backend, IReadOnlyList<int> ids)
    {
        int seq = ids.Count, d = DModel;
        float[] x = new float[(long)seq * d];
        float* emb = (float*)W("token_embd.weight").DataPointer;
        for (int s = 0; s < seq; s++) for (int c = 0; c < d; c++) x[s * d + c] = emb[(long)ids[s] * d + c];
        LayerNorm(x, seq, (float*)W("token_embd_norm.weight").DataPointer, (float*)W("token_embd_norm.bias").DataPointer);

        float[]? vFirst = null;
        for (int il = 0; il < NumLayers; il++)
        {
            TimeMix(backend, x, seq, il, ref vFirst);
            ChannelMix(backend, x, seq, il);
        }
        LayerNorm(x, seq, (float*)W("output_norm.weight").DataPointer, (float*)W("output_norm.bias").DataPointer);

        float* outW = (float*)W("output.weight").DataPointer;   // raw [vocab, d]
        float[] logits = new float[VocabSize];
        for (int v = 0; v < VocabSize; v++)
        {
            float acc = 0f; long wb = (long)v * d, xb = (long)(seq - 1) * d;
            for (int c = 0; c < d; c++) acc += x[xb + c] * outW[wb + c];
            logits[v] = acc;
        }
        return logits;
    }

    // xx[-1] is the carried last row from the previous call (zero at true sequence position 0 — prevRow starts
    // zeroed and ResetState() re-zeros it).
    private float[] ShiftDiff(float[] xx, int t, float[] prevRow) => RwkvOps.ShiftDiff(xx, t, DModel, prevRow);

    private void TimeMix(IBackend backend, float[] x, int t, int il, ref float[]? vFirst)
    {
        int d = DModel, H = NumHeads, N = HeadSize;
        string p = $"blk.{il}.";
        float[] xx = (float[])x.Clone();
        LayerNorm(xx, t, (float*)W(p + "attn_norm.weight").DataPointer, (float*)W(p + "attn_norm.bias").DataPointer);
        float[] sx = ShiftDiff(xx, t, _timeMixPrevRow[il]);

        // Fused token-shift lerp for the 6 components r,w,k,v,a,g: xc = xx + sx * lerp_fused[c].
        float* lerp = (float*)W(p + "time_mix_lerp_fused.weight").DataPointer;   // [d, (1,1,) 6] → component c at c*d
        float[] Mix(int comp)
        {
            float[] o = new float[(long)t * d];
            long cb = (long)comp * d;
            for (int s = 0; s < t; s++) for (int c = 0; c < d; c++) o[s * d + c] = xx[s * d + c] + sx[s * d + c] * lerp[cb + c];
            return o;
        }
        float[] xr = Mix(0), xw = Mix(1), xk = Mix(2), xv = Mix(3), xa = Mix(4), xg = Mix(5);

        float[] r = Lin(backend, xr, t, d, p + "time_mix_receptance.weight");
        // decay: w = exp(-0.606531 · sigmoid(w0 + tanh(xw·w1)·w2))
        float[] wl = Lin(backend, xw, t, d, p + "time_mix_w1.weight");
        for (long n = 0; n < wl.Length; n++) wl[n] = MathF.Tanh(wl[n]);
        float[] w = Lin(backend, wl, t, (int)W(p + "time_mix_w1.weight").Shape[0], p + "time_mix_w2.weight");
        float* w0 = (float*)W(p + "time_mix_w0.weight").DataPointer;
        for (int s = 0; s < t; s++) for (int c = 0; c < d; c++) { float wv = w[s * d + c] + w0[c]; w[s * d + c] = MathF.Exp(-0.606531f * (1f / (1f + MathF.Exp(-wv)))); }

        float[] k = Lin(backend, xk, t, d, p + "time_mix_key.weight");
        float[] v0v = Lin(backend, xv, t, d, p + "time_mix_value.weight");
        if (vFirst is null) vFirst = (float[])v0v.Clone();
        else
        {
            // value residual mix: v = v + (v_first - v) · sigmoid(v0 + (xv·v1)·v2)
            float[] vl = Lin(backend, xv, t, d, p + "time_mix_v1.weight");
            float[] vm = Lin(backend, vl, t, (int)W(p + "time_mix_v1.weight").Shape[0], p + "time_mix_v2.weight");
            float* v0 = (float*)W(p + "time_mix_v0.weight").DataPointer;
            for (long n = 0; n < v0v.Length; n++) { float mixv = 1f / (1f + MathF.Exp(-(vm[n] + v0[n % d]))); v0v[n] += (vFirst[n] - v0v[n]) * mixv; }
        }
        float[] v = v0v;

        // gate: g = sigmoid(xg·g1)·g2
        float[] gl = Lin(backend, xg, t, d, p + "time_mix_g1.weight");
        for (long n = 0; n < gl.Length; n++) gl[n] = 1f / (1f + MathF.Exp(-gl[n]));
        float[] g = Lin(backend, gl, t, (int)W(p + "time_mix_g1.weight").Shape[0], p + "time_mix_g2.weight");

        // a (iclr): sigmoid(a0 + (xa·a1)·a2)
        float[] al = Lin(backend, xa, t, d, p + "time_mix_a1.weight");
        float[] a = Lin(backend, al, t, (int)W(p + "time_mix_a1.weight").Shape[0], p + "time_mix_a2.weight");
        float* a0 = (float*)W(p + "time_mix_a0.weight").DataPointer;
        for (int s = 0; s < t; s++) for (int c = 0; c < d; c++) { float vv = a[s * d + c] + a0[c]; a[s * d + c] = 1f / (1f + MathF.Exp(-vv)); }

        // kk = L2-norm-per-head(k · k_k); k = k + (k·k_a)·(a−1).
        float* kk_w = (float*)W(p + "time_mix_k_k.weight").DataPointer;
        float* ka_w = (float*)W(p + "time_mix_k_a.weight").DataPointer;
        float[] kk = new float[(long)t * d];
        for (int s = 0; s < t; s++)
            for (int h = 0; h < H; h++)
            {
                long hb = (long)s * d + (long)h * N;
                double nrm = 0; for (int i = 0; i < N; i++) { float kv = k[hb + i] * kk_w[h * N + i]; kk[hb + i] = kv; nrm += (double)kv * kv; }
                float inv = (float)(1.0 / Math.Sqrt(nrm + 1e-12));
                for (int i = 0; i < N; i++) kk[hb + i] *= inv;
            }
        for (int s = 0; s < t; s++) for (int c = 0; c < d; c++) { float ka = k[s * d + c] * ka_w[c]; k[s * d + c] += ka * (a[s * d + c] - 1f); }

        // WKV7 delta-rule recurrence. Op inputs: r, w, k, v, aIn = −kk, bIn = kk·a. State S[i,j] per head,
        // carried across calls — zero only at true position 0.
        // for i: sa = Σ_j aIn[j]·S_prev[i,j]; for j: S[i,j] = S_prev[i,j]·w[j] + v[i]·k[j] + sa·bIn[j]; out[i] += S[i,j]·r[j]
        float[] state = _wkvState[il];
        float[] outv = new float[(long)t * d];
        for (int s = 0; s < t; s++)
            for (int h = 0; h < H; h++)
            {
                long hb = (long)s * d + (long)h * N, sb = (long)h * N * N;
                for (int i = 0; i < N; i++)
                {
                    float vi = v[hb + i];
                    float sa = 0f;
                    long si = sb + (long)i * N;
                    for (int j = 0; j < N; j++) sa += (-kk[hb + j]) * state[si + j];
                    float res = 0f;
                    for (int j = 0; j < N; j++)
                    {
                        float bIn = kk[hb + j] * a[hb + j];
                        float ns = state[si + j] * w[hb + j] + vi * k[hb + j] + sa * bIn;
                        state[si + j] = ns;
                        res += ns * r[hb + j];
                    }
                    outv[hb + i] = res;
                }
            }

        // per-head GroupNorm (eps 64e-5) · ln_w + ln_b.
        float* lnw = (float*)W(p + "time_mix_ln.weight").DataPointer; float* lnb = (float*)W(p + "time_mix_ln.bias").DataPointer;
        for (int s = 0; s < t; s++)
            for (int h = 0; h < H; h++)
            {
                long hb = (long)s * d + (long)h * N;
                double mean = 0; for (int i = 0; i < N; i++) mean += outv[hb + i]; mean /= N;
                double var = 0; for (int i = 0; i < N; i++) { double dd = outv[hb + i] - mean; var += dd * dd; } var /= N;
                float inv = (float)(1.0 / Math.Sqrt(var + 64e-5));
                for (int i = 0; i < N; i++) { int c = h * N + i; outv[hb + i] = (float)((outv[hb + i] - mean) * inv) * lnw[c] + lnb[c]; }
            }

        // r·k bonus: rk[h] = Σ_i k[h,i]·r[h,i]·r_k[h,i]; out[h,i] += v[h,i]·rk[h]. Then gate. Then out_proj.
        float* rk_w = (float*)W(p + "time_mix_r_k.weight").DataPointer;
        for (int s = 0; s < t; s++)
            for (int h = 0; h < H; h++)
            {
                long hb = (long)s * d + (long)h * N;
                float rk = 0f; for (int i = 0; i < N; i++) rk += k[hb + i] * r[hb + i] * rk_w[h * N + i];
                for (int i = 0; i < N; i++) outv[hb + i] = (outv[hb + i] + v[hb + i] * rk) * g[hb + i];
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
        float* lk = (float*)W(p + "channel_mix_lerp_k.weight").DataPointer;
        float[] kx = new float[(long)t * d];
        for (int s = 0; s < t; s++) for (int c = 0; c < d; c++) kx[s * d + c] = xx[s * d + c] + sx[s * d + c] * lk[c];
        float[] kk = Lin(backend, kx, t, d, p + "channel_mix_key.weight");   // [t, ffn]
        for (long n = 0; n < kk.Length; n++) { float relu = MathF.Max(kk[n], 0f); kk[n] = relu * relu; }   // squared ReLU
        float[] kv = Lin(backend, kk, t, Ffn, p + "channel_mix_value.weight");   // [t, d]
        for (long n = 0; n < (long)t * d; n++) x[n] += kv[n];   // no receptance gate in RWKV-7 channel-mix
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _handle.Dispose();
    }
}
