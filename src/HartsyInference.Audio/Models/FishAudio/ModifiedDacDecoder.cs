using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Code-to-waveform path of fish-speech's ModifiedDAC (<c>DownsampleResidualVectorQuantize.decode</c> followed
/// by the causal DAC <c>Decoder</c>): per-codebook lookup + 1×1 out-projection summed into the latent, the post-module
/// window-limited transformer, the causal ConvTranspose/ConvNeXt upsampler, then Snake / causal transposed-conv stages
/// with residual units and a final tanh. Reference-audio encoding is not part of this class.</summary>
public sealed unsafe class ModifiedDacDecoder : IDisposable
{
    private readonly ModifiedDacConfig _cfg;
    private Tensor[] _codebook = [], _projW = [], _projB = [];
    private TransformerLayer[] _layers = [];
    private Tensor? _postNorm;
    private Tensor[] _upW = [], _upB = [];
    private ConvNeXt[] _upNext = [];
    private Tensor? _stemW, _stemB, _finalAlpha, _finalW, _finalB;
    private DecoderStage[] _stages = [];
    private int _disposed;

    private sealed record TransformerLayer(Tensor AttnNorm, Tensor Wqkv, Tensor Wo, Tensor AttnScale,
        Tensor FfnNorm, Tensor W1, Tensor W3, Tensor W2, Tensor FfnScale);

    private sealed record ConvNeXt(Tensor DwW, Tensor DwB, Tensor NormW, Tensor NormB, Tensor Pw1W, Tensor Pw1B,
        Tensor Pw2W, Tensor Pw2B, Tensor Gamma);

    private sealed record DecoderStage(Tensor Alpha, Tensor UpW, Tensor UpB, Unit[] Units);

    private sealed record Unit(Tensor Alpha1, Tensor Conv1W, Tensor Conv1B, Tensor Alpha2, Tensor Conv2W, Tensor Conv2B);

    public ModifiedDacDecoder(ModifiedDacConfig cfg) { _cfg = cfg; }

    public int SampleRate => _cfg.SampleRate;

    private static Tensor F(IReadOnlyDictionary<string, Tensor> w, string key) => WhisperOps.EnsureF32(w[key]);

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        int books = _cfg.TotalCodebooks;
        _codebook = new Tensor[books]; _projW = new Tensor[books]; _projB = new Tensor[books];
        for (int i = 0; i < books; i++)
        {
            string p = i == 0 ? "quantizer.semantic_quantizer.quantizers.0" : $"quantizer.quantizer.quantizers.{i - 1}";
            _codebook[i] = F(w, $"{p}.codebook.weight");
            _projW[i] = WeightNorm.Compose(w, $"{p}.out_proj");
            _projB[i] = F(w, $"{p}.out_proj.bias");
        }

        _layers = new TransformerLayer[_cfg.TransformerLayers];
        for (int i = 0; i < _layers.Length; i++)
        {
            string p = $"quantizer.post_module.layers.{i}";
            _layers[i] = new TransformerLayer(F(w, $"{p}.attention_norm.weight"), F(w, $"{p}.attention.wqkv.weight"),
                F(w, $"{p}.attention.wo.weight"), F(w, $"{p}.attention_layer_scale.gamma"), F(w, $"{p}.ffn_norm.weight"),
                F(w, $"{p}.feed_forward.w1.weight"), F(w, $"{p}.feed_forward.w3.weight"), F(w, $"{p}.feed_forward.w2.weight"),
                F(w, $"{p}.ffn_layer_scale.gamma"));
        }
        _postNorm = F(w, "quantizer.post_module.norm.weight");

        int stages = _cfg.UpsampleFactors.Length;
        _upW = new Tensor[stages]; _upB = new Tensor[stages]; _upNext = new ConvNeXt[stages];
        for (int i = 0; i < stages; i++)
        {
            string p = $"quantizer.upsample.{i}";
            _upW[i] = F(w, $"{p}.0.conv.weight"); _upB[i] = F(w, $"{p}.0.conv.bias");
            _upNext[i] = new ConvNeXt(F(w, $"{p}.1.dwconv.conv.weight"), F(w, $"{p}.1.dwconv.conv.bias"),
                F(w, $"{p}.1.norm.weight"), F(w, $"{p}.1.norm.bias"), F(w, $"{p}.1.pwconv1.weight"), F(w, $"{p}.1.pwconv1.bias"),
                F(w, $"{p}.1.pwconv2.weight"), F(w, $"{p}.1.pwconv2.bias"), F(w, $"{p}.1.gamma"));
        }

        _stemW = WeightNorm.Compose(w, "decoder.model.0.conv"); _stemB = F(w, "decoder.model.0.conv.bias");
        _stages = new DecoderStage[_cfg.DecoderRates.Length];
        for (int i = 0; i < _stages.Length; i++)
        {
            string p = $"decoder.model.{i + 1}.block";
            Unit[] units = new Unit[_cfg.ResidualDilations.Length];
            for (int j = 0; j < units.Length; j++)
            {
                string u = $"{p}.{j + 2}.block";
                units[j] = new Unit(F(w, $"{u}.0.alpha"), WeightNorm.Compose(w, $"{u}.1.conv"), F(w, $"{u}.1.conv.bias"),
                    F(w, $"{u}.2.alpha"), WeightNorm.Compose(w, $"{u}.3.conv"), F(w, $"{u}.3.conv.bias"));
            }
            _stages[i] = new DecoderStage(F(w, $"{p}.0.alpha"), WeightNorm.Compose(w, $"{p}.1.conv"), F(w, $"{p}.1.conv.bias"), units);
        }
        int last = _stages.Length + 1;
        _finalAlpha = F(w, $"decoder.model.{last}.alpha");
        _finalW = WeightNorm.Compose(w, $"decoder.model.{last + 1}.conv");
        _finalB = F(w, $"decoder.model.{last + 1}.conv.bias");
    }

    /// <summary>Decodes <c>[1 + numResidual, T]</c> codes (row 0 = semantic) to mono PCM in [-1, 1],
    /// <c>T × SamplesPerFrame</c> samples long. Out-of-range codes are clamped, as upstream does.</summary>
    public float[] Decode(IBackend backend, int[,] codes, int t, Action<string, float[]>? tap = null)
    {
        int books = _cfg.TotalCodebooks, d = _cfg.LatentDim;
        if (codes.GetLength(0) != books) throw new ArgumentException($"codes must have {books} rows.", nameof(codes));
        if (t <= 0) throw new ArgumentException("t must be positive.", nameof(t));

        // Σ out_proj_i(codebook_i[code_i]) → [T, D]
        float[] z = new float[t * d];
        for (int i = 0; i < books; i++)
        {
            int size = i == 0 ? _cfg.SemanticCodebookSize : _cfg.ResidualCodebookSize, cb = _cfg.CodebookDim;
            float* book = (float*)_codebook[i].DataPointer, pw = (float*)_projW[i].DataPointer, pb = (float*)_projB[i].DataPointer;
            for (int j = 0; j < t; j++)
            {
                float* vec = book + (long)Math.Clamp(codes[i, j], 0, size - 1) * cb;
                for (int c = 0; c < d; c++)
                {
                    float acc = pb[c];
                    float* row = pw + (long)c * cb;
                    for (int k = 0; k < cb; k++) acc += row[k] * vec[k];
                    z[j * d + c] += acc;
                }
            }
        }

        tap?.Invoke("sum", z);
        float[] h = PostTransformer(backend, z, t);
        tap?.Invoke("post", h);

        // channels-first [1, D, T] for the conv stack
        Tensor x = new(new TensorShape(1, d, t), DType.F32);
        float* xp = (float*)x.DataPointer;
        for (int j = 0; j < t; j++) for (int c = 0; c < d; c++) xp[c * t + j] = h[j * d + c];

        int curT = t;
        for (int i = 0; i < _upW.Length; i++)
        {
            int s = _cfg.UpsampleFactors[_cfg.UpsampleFactors.Length - 1 - i];   // reversed(downsample_factor)
            x = CausalConvTranspose(backend, x, _upW[i], _upB[i], s, kernel: s);
            curT *= s;
            tap?.Invoke($"up{i}_conv", ToHost(x));
            x = ConvNeXtBlock(backend, x, _upNext[i], curT);
            tap?.Invoke($"up{i}", ToHost(x));
        }

        x = CausalConv(backend, x, _stemW!, _stemB, 1, curT);
        tap?.Invoke("stem", ToHost(x));
        int dim = _cfg.DecoderDim;
        for (int i = 0; i < _stages.Length; i++)
        {
            DecoderStage st = _stages[i];
            int stride = _cfg.DecoderRates[i];
            x = SnakeOp(backend, x, st.Alpha);
            x = CausalConvTranspose(backend, x, st.UpW, st.UpB, stride, kernel: 2 * stride);
            curT *= stride; dim /= 2;
            tap?.Invoke($"stage{i}_up", ToHost(x));
            for (int j = 0; j < st.Units.Length; j++) x = ResidualUnit(backend, x, st.Units[j], _cfg.ResidualDilations[j], curT);
            tap?.Invoke($"stage{i}", ToHost(x));
        }
        x = SnakeOp(backend, x, _finalAlpha!);
        x = CausalConv(backend, x, _finalW!, _finalB, 1, curT);
        float[] audio = new float[curT];
        float* ap = (float*)x.DataPointer;
        for (int j = 0; j < curT; j++) audio[j] = MathF.Tanh(ap[j]);
        x.Dispose();
        return audio;
    }

    // ---- window-limited causal transformer (post_module) --------------------------------------------------------

    private float[] PostTransformer(IBackend backend, float[] z, int t)
    {
        int d = _cfg.LatentDim, heads = _cfg.TransformerHeads, hd = _cfg.TransformerHeadDim, inter = _cfg.TransformerIntermediate;
        float[] x = z;
        float[,] rope = RopeTable(t, hd, _cfg.TransformerRopeBase);
        foreach (TransformerLayer layer in _layers)
        {
            // h = x + γ_a ⊙ attention(norm(x))
            float[] normed = RmsNorm(backend, x, layer.AttnNorm, t, d);
            float[] qkv = Linear(backend, normed, layer.Wqkv, t, d, 3 * d);
            float[] attn = new float[t * d];
            float scale = 1f / MathF.Sqrt(hd);
            int window = _cfg.TransformerWindow;
            for (int hh = 0; hh < heads; hh++)
            {
                float[] q = new float[t * hd], k = new float[t * hd];
                for (int i = 0; i < t; i++)
                    for (int e = 0; e < hd; e++)
                    {
                        q[i * hd + e] = qkv[i * 3 * d + hh * hd + e];
                        k[i * hd + e] = qkv[i * 3 * d + d + hh * hd + e];
                    }
                ApplyRopeInterleaved(q, rope, t, hd); ApplyRopeInterleaved(k, rope, t, hd);
                float[] sc = new float[t];
                for (int i = 0; i < t; i++)
                {
                    int lo = Math.Max(0, i - window + 1);
                    float max = float.NegativeInfinity;
                    for (int j = lo; j <= i; j++)
                    {
                        float s = 0f;
                        for (int e = 0; e < hd; e++) s += q[i * hd + e] * k[j * hd + e];
                        sc[j] = s * scale; max = MathF.Max(max, sc[j]);
                    }
                    float sum = 0f;
                    for (int j = lo; j <= i; j++) { sc[j] = MathF.Exp(sc[j] - max); sum += sc[j]; }
                    for (int e = 0; e < hd; e++)
                    {
                        float o = 0f;
                        for (int j = lo; j <= i; j++) o += sc[j] * qkv[j * 3 * d + 2 * d + hh * hd + e];
                        attn[i * d + hh * hd + e] = o / sum;
                    }
                }
            }
            float[] proj = Linear(backend, attn, layer.Wo, t, d, d);
            float* ga = (float*)layer.AttnScale.DataPointer;
            float[] hres = new float[t * d];
            for (int i = 0; i < t; i++) for (int c = 0; c < d; c++) hres[i * d + c] = x[i * d + c] + ga[c] * proj[i * d + c];

            // out = h + γ_f ⊙ w2(silu(w1(norm(h))) ⊙ w3(norm(h)))
            float[] n2 = RmsNorm(backend, hres, layer.FfnNorm, t, d);
            float[] a = Linear(backend, n2, layer.W1, t, d, inter), b = Linear(backend, n2, layer.W3, t, d, inter);
            for (int i = 0; i < a.Length; i++) a[i] = a[i] / (1f + MathF.Exp(-a[i])) * b[i];
            float[] ff = Linear(backend, a, layer.W2, t, inter, d);
            float* gf = (float*)layer.FfnScale.DataPointer;
            float[] next = new float[t * d];
            for (int i = 0; i < t; i++) for (int c = 0; c < d; c++) next[i * d + c] = hres[i * d + c] + gf[c] * ff[i * d + c];
            x = next;
        }
        return RmsNorm(backend, x, _postNorm!, t, d);
    }

    private float[] RmsNorm(IBackend backend, float[] x, Tensor weight, int t, int d)
    {
        using Tensor inp = FromHost(x, 1, t, d);
        using Tensor outT = new(new TensorShape(1, t, d), DType.F32);
        backend.RmsNorm(outT, inp, weight, _cfg.TransformerNormEps);
        return ToHost(outT);
    }

    private static float[] Linear(IBackend backend, float[] x, Tensor weight, int t, int k, int n, Tensor? bias = null)
    {
        using Tensor inp = FromHost(x, 1, t, k);
        using Tensor outT = new(new TensorShape(1, t, n), DType.F32);
        backend.Linear(outT, inp, weight, bias);
        return ToHost(outT);
    }

    private static float[,] RopeTable(int t, int hd, float baseTheta)
    {
        float[,] table = new float[t, hd];   // [pos, 2i] = cos, [pos, 2i+1] = sin
        for (int p = 0; p < t; p++)
            for (int i = 0; i < hd / 2; i++)
            {
                double f = 1.0 / Math.Pow(baseTheta, 2.0 * i / hd), a = p * f;
                table[p, 2 * i] = (float)Math.Cos(a); table[p, 2 * i + 1] = (float)Math.Sin(a);
            }
        return table;
    }

    // fish-speech apply_rotary_emb: adjacent pairs (x[2i], x[2i+1]) rotate by position-dependent angle i.
    private static void ApplyRopeInterleaved(float[] v, float[,] table, int t, int hd)
    {
        for (int p = 0; p < t; p++)
            for (int i = 0; i < hd / 2; i++)
            {
                float c = table[p, 2 * i], s = table[p, 2 * i + 1];
                float x0 = v[p * hd + 2 * i], x1 = v[p * hd + 2 * i + 1];
                v[p * hd + 2 * i] = x0 * c - x1 * s;
                v[p * hd + 2 * i + 1] = x1 * c + x0 * s;
            }
    }

    // ---- conv stack ---------------------------------------------------------------------------------------------

    // CausalConvNet: left-pad (k-1)·dilation, no right pad (stride 1 needs no extra alignment padding).
    private static Tensor CausalConv(IBackend backend, Tensor x, Tensor w, Tensor? b, int dilation, int t, int groups = 1)
    {
        int k = (int)w.Shape[2], cOut = (int)w.Shape[0];
        Tensor o = new(new TensorShape(1, cOut, t), DType.F32);
        backend.Conv1d(o, x, w, b, 1, (k - 1) * dilation, 0, dilation, groups);
        x.Dispose();
        return o;
    }

    // CausalTransConvNet: transposed conv then drop the trailing k-stride samples → exactly stride× the input length.
    private static Tensor CausalConvTranspose(IBackend backend, Tensor x, Tensor w, Tensor b, int stride, int kernel)
    {
        int cOut = (int)w.Shape[1], tIn = (int)x.Shape[2], tOut = tIn * stride;
        Tensor o = new(new TensorShape(1, cOut, tOut), DType.F32);
        backend.ConvTranspose1d(o, x, w, b, stride, 0, kernel - stride, 1, 1);
        x.Dispose();
        return o;
    }

    private static Tensor SnakeOp(IBackend backend, Tensor x, Tensor alpha)
    {
        Tensor o = new(x.Shape, DType.F32);
        backend.Snake(o, x, alpha, null);
        x.Dispose();
        return o;
    }

    private Tensor ResidualUnit(IBackend backend, Tensor x, Unit u, int dilation, int t)
    {
        Tensor y = SnakeOp(backend, CloneTensor(x), u.Alpha1);
        y = CausalConv(backend, y, u.Conv1W, u.Conv1B, dilation, t);
        y = SnakeOp(backend, y, u.Alpha2);
        y = CausalConv(backend, y, u.Conv2W, u.Conv2B, 1, t);
        float* xp = (float*)x.DataPointer, yp = (float*)y.DataPointer;
        for (long i = 0; i < y.ElementCount; i++) yp[i] += xp[i];
        x.Dispose();
        return y;
    }

    // ConvNeXtBlock: causal depthwise k7 → LayerNorm(eps 1e-6) → Linear(4×) → GELU → Linear → γ → + input.
    private Tensor ConvNeXtBlock(IBackend backend, Tensor x, ConvNeXt b, int t)
    {
        int d = _cfg.LatentDim;
        Tensor dw = CausalConv(backend, CloneTensor(x), b.DwW, b.DwB, 1, t, groups: d);
        float[] h = new float[t * d];
        float* dp = (float*)dw.DataPointer;
        for (int c = 0; c < d; c++) for (int j = 0; j < t; j++) h[j * d + c] = dp[c * t + j];
        dw.Dispose();

        using (Tensor inp = FromHost(h, 1, t, d))
        using (Tensor normed = new(new TensorShape(1, t, d), DType.F32))
        {
            backend.LayerNorm(normed, inp, b.NormW, b.NormB, 1e-6f);
            h = ToHost(normed);
        }
        int hidden = (int)b.Pw1W.Shape[0];
        float[] a = Linear(backend, h, b.Pw1W, t, d, hidden, b.Pw1B);
        for (int i = 0; i < a.Length; i++) a[i] = 0.5f * a[i] * (1f + (float)Erf(a[i] / Math.Sqrt(2.0)));   // exact GELU
        float[] o = Linear(backend, a, b.Pw2W, t, hidden, d, b.Pw2B);
        float* g = (float*)b.Gamma.DataPointer;
        float* xp = (float*)x.DataPointer;
        for (int c = 0; c < d; c++) for (int j = 0; j < t; j++) xp[c * t + j] += g[c] * o[j * d + c];
        return x;
    }

    private static double Erf(double x)
    {
        // Abramowitz-Stegun 7.1.26 is too coarse for parity; use a series/continued-fraction split.
        double ax = Math.Abs(x);
        double r;
        if (ax < 2.5)
        {
            double sum = ax, term = ax;
            for (int n = 1; n < 60; n++) { term *= -ax * ax / n; sum += term / (2 * n + 1); }
            r = 2.0 / Math.Sqrt(Math.PI) * sum;
        }
        else
        {
            double f = 0;
            for (int n = 60; n >= 1; n--) f = n / 2.0 / (ax + f);
            r = 1.0 - Math.Exp(-ax * ax) / Math.Sqrt(Math.PI) / (ax + f);
        }
        return x < 0 ? -r : r;
    }

    private static Tensor CloneTensor(Tensor x)
    {
        Tensor c = new(x.Shape, DType.F32);
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)c.DataPointer, x.ElementCount * 4, x.ElementCount * 4);
        return c;
    }

    private static Tensor FromHost(float[] v, int b, int t, int d)
    {
        Tensor x = new(new TensorShape(b, t, d), DType.F32);
        v.AsSpan().CopyTo(new Span<float>((void*)x.DataPointer, v.Length));
        return x;
    }

    private static float[] ToHost(Tensor x) => new ReadOnlySpan<float>((void*)x.DataPointer, (int)x.ElementCount).ToArray();

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
