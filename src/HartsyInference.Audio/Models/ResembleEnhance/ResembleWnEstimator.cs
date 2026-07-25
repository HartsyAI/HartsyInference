using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ResembleEnhance;

/// <summary>Resemble-enhance CFM velocity network (upstream <c>enhancer/lcfm/wn.py</c> + the CFM time embedding
/// from <c>cfm.py</c>): <c>start</c> 1×1 conv into 512 hidden, then 30 DiffWave-style layers — each adds a
/// per-layer <c>gconv</c> projection of the sinusoidal time embedding BEFORE its dilated <c>dconv</c>
/// (512→1024, k3, dilation 2^(i%5)), adds a per-layer <c>lconv</c> projection of the InstanceNorm'd
/// conditioning mel, gates tanh·sigmoid, and projects via <c>out</c> (512→1024) whose halves are the residual
/// (added to the layer input and scaled by 1/√2) and the skip. Skips sum, scale by 1/√30, and <c>end</c>
/// projects to the 64-dim latent velocity. The time embedding is <c>[sin(t·10^p), cos(t·10^p)]</c> with
/// <c>p = linspace(0, 4, 64)</c>.</summary>
public sealed unsafe class ResembleWnEstimator
{
    private readonly int _hidden, _layers, _kernel, _cycle, _latent, _melDim, _timeDim;
    private Tensor? _startW, _startB, _endW, _endB;
    private readonly Tensor?[] _gW, _gB, _dW, _dB, _lW, _lB, _oW, _oB;

    public ResembleWnEstimator(ResembleEnhanceConfig cfg, int melDim)
    {
        _hidden = cfg.WnHidden;
        _layers = cfg.WnLayers;
        _kernel = cfg.WnKernel;
        _cycle = cfg.WnDilationCycle;
        _latent = cfg.LatentDim;
        _melDim = melDim;
        _timeDim = cfg.TimeEmbDim;
        _gW = new Tensor?[_layers];
        _gB = new Tensor?[_layers];
        _dW = new Tensor?[_layers];
        _dB = new Tensor?[_layers];
        _lW = new Tensor?[_layers];
        _lB = new Tensor?[_layers];
        _oW = new Tensor?[_layers];
        _oB = new Tensor?[_layers];
    }

    public void LoadWeights(ResembleWeightReader r, string prefix = "lcfm.cfm.net")
    {
        _startW = r.F32($"{prefix}.start.weight");
        _startB = r.F32($"{prefix}.start.bias");
        _endW = r.F32($"{prefix}.end.weight");
        _endB = r.F32($"{prefix}.end.bias");
        for (int i = 0; i < _layers; i++)
        {
            _gW[i] = r.F32($"{prefix}.layers.{i}.gconv.weight");
            _gB[i] = r.F32($"{prefix}.layers.{i}.gconv.bias");
            _dW[i] = r.F32($"{prefix}.layers.{i}.dconv.weight");
            _dB[i] = r.F32($"{prefix}.layers.{i}.dconv.bias");
            _lW[i] = r.F32($"{prefix}.layers.{i}.lconv.weight");
            _lB[i] = r.F32($"{prefix}.layers.{i}.lconv.bias");
            _oW[i] = r.F32($"{prefix}.layers.{i}.out.weight");
            _oB[i] = r.F32($"{prefix}.layers.{i}.out.bias");
        }
    }

    /// <summary>Velocity estimate at flow time <paramref name="t"/> (clamped to [0,1] like upstream
    /// <c>_to_v</c>). <paramref name="x"/> is the latent <c>[1, latent, T]</c>; <paramref name="cond"/> the
    /// conditioning mel <c>[1, melDim, T]</c>.</summary>
    public Tensor Estimate(IBackend backend, Tensor x, Tensor cond, float t)
    {
        int h = _hidden, tl = (int)x.Shape[2];
        Tensor z = new(new TensorShape(1, h, tl), DType.F32);
        backend.Conv1d(z, x, _startW!, _startB, 1, 0, 0, 1, 1);
        float* zp = (float*)z.DataPointer;

        // Sinusoidal time embedding g[128], projected per layer through gconv.
        float tc = Math.Clamp(t, 0f, 1f);
        float[] emb = TimeEmbedding(tc, _timeDim);

        // Local conditioning: parameter-free InstanceNorm over the mel, projected per layer through lconv.
        Tensor condNorm = InstanceNorm(cond, _melDim, tl);

        Tensor skipAcc = new(new TensorShape(1, h, tl), DType.F32);
        float* sa = (float*)skipAcc.DataPointer;

        float invSqrt2 = 1f / MathF.Sqrt(2f);
        for (int i = 0; i < _layers; i++)
        {
            // z_in = z + gconv(g), broadcast over time.
            Tensor zin = new(new TensorShape(1, h, tl), DType.F32);
            float* zinp = (float*)zin.DataPointer;
            float* gw = (float*)_gW[i]!.DataPointer;   // [h, timeDim, 1]
            float* gb = (float*)_gB[i]!.DataPointer;
            for (int c = 0; c < h; c++)
            {
                float acc = gb[c];
                long row = (long)c * _timeDim;
                for (int d = 0; d < _timeDim; d++)
                {
                    acc += gw[row + d] * emb[d];
                }
                long off = (long)c * tl;
                for (int j = 0; j < tl; j++)
                {
                    zinp[off + j] = zp[off + j] + acc;
                }
            }

            int dilation = 1 << (i % _cycle);
            int pad = dilation * (_kernel - 1) / 2;
            Tensor gate = new(new TensorShape(1, 2 * h, tl), DType.F32);
            backend.Conv1d(gate, zin, _dW[i]!, _dB[i], 1, pad, pad, dilation, 1);
            zin.Dispose();

            Tensor local = new(new TensorShape(1, 2 * h, tl), DType.F32);
            backend.Conv1d(local, condNorm, _lW[i]!, _lB[i], 1, 0, 0, 1, 1);

            // Fused tanh·sigmoid over the two halves.
            Tensor acts = new(new TensorShape(1, h, tl), DType.F32);
            float* ap = (float*)acts.DataPointer;
            float* dp = (float*)gate.DataPointer;
            float* lp = (float*)local.DataPointer;
            for (int c = 0; c < h; c++)
            {
                long offA = (long)c * tl;
                long offB = (long)(h + c) * tl;
                for (int j = 0; j < tl; j++)
                {
                    float a = dp[offA + j] + lp[offA + j];
                    float b = dp[offB + j] + lp[offB + j];
                    ap[offA + j] = MathF.Tanh(a) * (1f / (1f + MathF.Exp(-b)));
                }
            }
            gate.Dispose();
            local.Dispose();

            // out: 512→1024, chunked into residual (added to layer input, /√2) and skip.
            Tensor proj = new(new TensorShape(1, 2 * h, tl), DType.F32);
            backend.Conv1d(proj, acts, _oW[i]!, _oB[i], 1, 0, 0, 1, 1);
            acts.Dispose();
            float* pp = (float*)proj.DataPointer;
            for (int c = 0; c < h; c++)
            {
                long off = (long)c * tl;
                long offS = (long)(h + c) * tl;
                for (int j = 0; j < tl; j++)
                {
                    zp[off + j] = (pp[off + j] + zp[off + j]) * invSqrt2;
                    sa[off + j] += pp[offS + j];
                }
            }
            proj.Dispose();
        }
        condNorm.Dispose();
        z.Dispose();

        float inv = 1f / MathF.Sqrt(_layers);
        long total = (long)h * tl;
        for (long n = 0; n < total; n++)
        {
            sa[n] *= inv;
        }
        Tensor outT = new(new TensorShape(1, _latent, tl), DType.F32);
        backend.Conv1d(outT, skipAcc, _endW!, _endB, 1, 0, 0, 1, 1);
        skipAcc.Dispose();
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_startW, _startB, _endW, _endB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        Tensor?[][] g = [_gW, _gB, _dW, _dB, _lW, _lB, _oW, _oB];
        foreach (Tensor?[] arr in g) foreach (Tensor? t in arr) if (t is not null) yield return t;
    }

    /// <summary>Upstream <c>SinusodialTimeEmbedding</c>: <c>p = linspace(0, 4, dim/2)</c> inclusive;
    /// <c>[sin(t·10^p), cos(t·10^p)]</c>.</summary>
    private static float[] TimeEmbedding(float t, int dim)
    {
        float[] e = new float[dim];
        int half = dim / 2;
        for (int i = 0; i < half; i++)
        {
            float p = half > 1 ? 4f * i / (half - 1) : 0f;
            float freq = MathF.Pow(10f, p);
            e[i] = MathF.Sin(t * freq);
            e[half + i] = MathF.Cos(t * freq);
        }
        return e;
    }

    private static Tensor InstanceNorm(Tensor x, int ch, int t)
    {
        Tensor o = new(new TensorShape(1, ch, t), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)o.DataPointer;
        for (int c = 0; c < ch; c++)
        {
            double mean = 0;
            for (int j = 0; j < t; j++) mean += xp[(long)c * t + j];
            mean /= t;
            double var = 0;
            for (int j = 0; j < t; j++)
            {
                double d = xp[(long)c * t + j] - mean;
                var += d * d;
            }
            var /= t;
            float invs = (float)(1.0 / Math.Sqrt(var + 1e-5));
            for (int j = 0; j < t; j++)
            {
                op[(long)c * t + j] = (xp[(long)c * t + j] - (float)mean) * invs;
            }
        }
        return o;
    }
}
