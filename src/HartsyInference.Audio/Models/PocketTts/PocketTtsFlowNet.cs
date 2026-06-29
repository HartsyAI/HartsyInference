using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.PocketTts;

/// <summary>Pocket-TTS <c>flow_net</c> = SimpleMLPAdaLN (the MAR DiffLoss MLP), matching kyutai-labs/pocket-tts.
/// Per frame: <c>v = flow_net(c, s, t, x)</c> where c is the AR transformer hidden, (s,t) are the two flow
/// times, x is the current latent. Structure: <c>input_proj</c> (32->512), two <c>TimestepEmbedder</c>s
/// (sinusoid freqs -> Linear -> SiLU -> Linear -> a non-standard RMSNorm) averaged, <c>cond_embed</c>
/// (1024->512) added to form y, then 6 adaLN <c>ResBlock</c>s and a <c>FinalLayer</c> (512->32).
///
/// <para>Numerical care: the timestep RMSNorm uses UNBIASED variance with NO mean subtraction (eps 1e-5);
/// the ResBlock/FinalLayer LayerNorms use population variance (eps 1e-6); <c>modulate = x*(1+scale)+shift</c>;
/// activation is SiLU except the AdaLN gates. Verified by the Phase C parity test.</para></summary>
internal sealed unsafe class PocketTtsFlowNet
{
    private const int Cond = 1024, Model = 512, Latent = 32, Freq = 256, Half = 128, NBlocks = 6;
    private readonly string _p;

    private Tensor? _inW, _inB;            // input_proj 32->512
    private Tensor? _condW, _condB;        // cond_embed 1024->512
    // time_embed[2]: freqs[128], mlp0 (256->512), mlp2 (512->512), rms alpha[512]
    private readonly Tensor?[] _teFreqs = new Tensor?[2];
    private readonly Tensor?[] _te0W = new Tensor?[2], _te0B = new Tensor?[2];
    private readonly Tensor?[] _te2W = new Tensor?[2], _te2B = new Tensor?[2];
    private readonly Tensor?[] _teAlpha = new Tensor?[2];
    // res_blocks[6]
    private readonly Tensor?[] _rInLnW = new Tensor?[NBlocks], _rInLnB = new Tensor?[NBlocks];
    private readonly Tensor?[] _rM0W = new Tensor?[NBlocks], _rM0B = new Tensor?[NBlocks];
    private readonly Tensor?[] _rM2W = new Tensor?[NBlocks], _rM2B = new Tensor?[NBlocks];
    private readonly Tensor?[] _rAdaW = new Tensor?[NBlocks], _rAdaB = new Tensor?[NBlocks];
    // final_layer
    private Tensor? _fAdaW, _fAdaB, _fLinW, _fLinB;

    public PocketTtsFlowNet(string prefix = "flow_lm.flow_net") => _p = prefix;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _inW = G(w, $"{_p}.input_proj.weight"); _inB = G(w, $"{_p}.input_proj.bias");
        _condW = G(w, $"{_p}.cond_embed.weight"); _condB = G(w, $"{_p}.cond_embed.bias");
        for (int i = 0; i < 2; i++)
        {
            string p = $"{_p}.time_embed.{i}";
            _teFreqs[i] = G(w, $"{p}.freqs");
            _te0W[i] = G(w, $"{p}.mlp.0.weight"); _te0B[i] = G(w, $"{p}.mlp.0.bias");
            _te2W[i] = G(w, $"{p}.mlp.2.weight"); _te2B[i] = G(w, $"{p}.mlp.2.bias");
            _teAlpha[i] = G(w, $"{p}.mlp.3.alpha");
        }
        for (int i = 0; i < NBlocks; i++)
        {
            string p = $"{_p}.res_blocks.{i}";
            _rInLnW[i] = G(w, $"{p}.in_ln.weight"); _rInLnB[i] = G(w, $"{p}.in_ln.bias");
            _rM0W[i] = G(w, $"{p}.mlp.0.weight"); _rM0B[i] = G(w, $"{p}.mlp.0.bias");
            _rM2W[i] = G(w, $"{p}.mlp.2.weight"); _rM2B[i] = G(w, $"{p}.mlp.2.bias");
            _rAdaW[i] = G(w, $"{p}.adaLN_modulation.1.weight"); _rAdaB[i] = G(w, $"{p}.adaLN_modulation.1.bias");
        }
        _fAdaW = G(w, $"{_p}.final_layer.adaLN_modulation.1.weight"); _fAdaB = G(w, $"{_p}.final_layer.adaLN_modulation.1.bias");
        _fLinW = G(w, $"{_p}.final_layer.linear.weight"); _fLinB = G(w, $"{_p}.final_layer.linear.bias");
    }

    private static Tensor G(IReadOnlyDictionary<string, Tensor> w, string k) => WhisperOps.EnsureF32(w[k]);

    /// <summary>cond[Cond], s, t scalars, x[Latent] -> velocity[Latent]. Single frame (batch 1).</summary>
    public float[] Forward(float[] cond, float s, float t, float[] x)
    {
        float[] xh = Linear(x, _inW!, _inB!, Latent, Model);
        float[] te = TimeEmbed(0, s);
        float[] te1 = TimeEmbed(1, t);
        for (int i = 0; i < Model; i++) te[i] = (te[i] + te1[i]) * 0.5f;
        float[] c = Linear(cond, _condW!, _condB!, Cond, Model);
        float[] y = new float[Model];
        for (int i = 0; i < Model; i++) y[i] = te[i] + c[i];

        for (int blk = 0; blk < NBlocks; blk++)
            xh = ResBlock(blk, xh, y);

        return FinalLayer(xh, y);
    }

    private float[] TimeEmbed(int idx, float tval)
    {
        float* fr = (float*)_teFreqs[idx]!.DataPointer;
        float[] emb = new float[Freq];
        for (int k = 0; k < Half; k++)
        {
            float arg = tval * fr[k];
            emb[k] = MathF.Cos(arg);
            emb[Half + k] = MathF.Sin(arg);
        }
        float[] h = Linear(emb, _te0W[idx]!, _te0B[idx]!, Freq, Model);
        Silu(h);
        h = Linear(h, _te2W[idx]!, _te2B[idx]!, Model, Model);
        // non-standard RMSNorm: unbiased variance, no mean subtraction, eps 1e-5, scale by alpha.
        float* al = (float*)_teAlpha[idx]!.DataPointer;
        double mean = 0; for (int i = 0; i < Model; i++) mean += h[i]; mean /= Model;
        double ss = 0; for (int i = 0; i < Model; i++) { double d = h[i] - mean; ss += d * d; }
        double var = ss / (Model - 1);
        float inv = (float)(1.0 / Math.Sqrt(var + 1e-5));
        for (int i = 0; i < Model; i++) h[i] = h[i] * al[i] * inv;
        return h;
    }

    private float[] ResBlock(int blk, float[] x, float[] y)
    {
        float[] sy = (float[])y.Clone(); Silu(sy);
        float[] mod = Linear(sy, _rAdaW[blk]!, _rAdaB[blk]!, Model, 3 * Model);   // shift|scale|gate
        float[] ln = LayerNorm(x, _rInLnW[blk], _rInLnB[blk]);                    // affine, eps 1e-6
        float[] h = new float[Model];
        for (int i = 0; i < Model; i++) h[i] = ln[i] * (1f + mod[Model + i]) + mod[i];   // modulate
        float[] m = Linear(h, _rM0W[blk]!, _rM0B[blk]!, Model, Model);
        Silu(m);
        m = Linear(m, _rM2W[blk]!, _rM2B[blk]!, Model, Model);
        float[] outp = new float[Model];
        for (int i = 0; i < Model; i++) outp[i] = x[i] + mod[2 * Model + i] * m[i];       // gate
        return outp;
    }

    private float[] FinalLayer(float[] x, float[] y)
    {
        float[] sy = (float[])y.Clone(); Silu(sy);
        float[] mod = Linear(sy, _fAdaW!, _fAdaB!, Model, 2 * Model);             // shift|scale
        float[] ln = LayerNorm(x, null, null);                                   // no affine, eps 1e-6
        float[] h = new float[Model];
        for (int i = 0; i < Model; i++) h[i] = ln[i] * (1f + mod[Model + i]) + mod[i];
        return Linear(h, _fLinW!, _fLinB!, Model, Latent);
    }

    // y[o] = sum_i x[i]*W[o,i] + b[o]   (W stored [out,in])
    private static float[] Linear(float[] x, Tensor wt, Tensor? b, int inDim, int outDim)
    {
        float* wp = (float*)wt.DataPointer;
        float* bp = b is null ? null : (float*)b.DataPointer;
        float[] o = new float[outDim];
        for (int oi = 0; oi < outDim; oi++)
        {
            double acc = bp is null ? 0.0 : bp[oi];
            long wb = (long)oi * inDim;
            for (int i = 0; i < inDim; i++) acc += (double)x[i] * wp[wb + i];
            o[oi] = (float)acc;
        }
        return o;
    }

    private static void Silu(float[] x)
    {
        for (int i = 0; i < x.Length; i++) x[i] = x[i] / (1f + MathF.Exp(-x[i]));
    }

    // population-variance LayerNorm (eps 1e-6), optional affine.
    private static float[] LayerNorm(float[] x, Tensor? wt, Tensor? bt)
    {
        int n = x.Length;
        double mean = 0; for (int i = 0; i < n; i++) mean += x[i]; mean /= n;
        double ss = 0; for (int i = 0; i < n; i++) { double d = x[i] - mean; ss += d * d; }
        float inv = (float)(1.0 / Math.Sqrt(ss / n + 1e-6));
        float* wp = wt is null ? null : (float*)wt.DataPointer;
        float* bp = bt is null ? null : (float*)bt.DataPointer;
        float[] o = new float[n];
        for (int i = 0; i < n; i++)
        {
            float v = (float)((x[i] - mean) * inv);
            o[i] = wp is null ? v : v * wp[i] + bp![i];
        }
        return o;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all =
        [
            _inW, _inB, _condW, _condB, _fAdaW, _fAdaB, _fLinW, _fLinB,
            .. _teFreqs, .. _te0W, .. _te0B, .. _te2W, .. _te2B, .. _teAlpha,
            .. _rInLnW, .. _rInLnB, .. _rM0W, .. _rM0B, .. _rM2W, .. _rM2B, .. _rAdaW, .. _rAdaB,
        ];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
