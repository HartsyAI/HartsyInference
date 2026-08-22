using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Vits;

/// <summary>VITS WaveNet residual stack (the <c>WN</c> module shared by the flow's coupling layers and the posterior encoder): N dilated gated layers — <c>in_layer</c> (dilated Conv1d → 2·hidden) → fused tanh·sigmoid gate → <c>res_skip_layer</c>; residual feeds the next layer, skips accumulate; optional speaker conditioning (<c>g</c>) is projected once and added per layer.</summary>
public sealed unsafe class VitsWaveNet
{
    private readonly int _hidden, _layers, _kernel, _dilationRate;
    private readonly Tensor?[] _inW, _inB, _rsW, _rsB;
    private Tensor? _condW, _condB;     // cond_layer (multispeaker g), optional

    public VitsWaveNet(int hidden, int kernel, int dilationRate, int layers)
    {
        _hidden = hidden; _kernel = kernel; _dilationRate = dilationRate; _layers = layers;
        _inW = new Tensor?[layers]; _inB = new Tensor?[layers];
        _rsW = new Tensor?[layers]; _rsB = new Tensor?[layers];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        for (int i = 0; i < _layers; i++)
        {
            _inW[i] = VitsWeights.Conv(w, $"{prefix}.in_layers.{i}");
            _inB[i] = VitsWeights.Bias(w, $"{prefix}.in_layers.{i}");
            _rsW[i] = VitsWeights.Conv(w, $"{prefix}.res_skip_layers.{i}");
            _rsB[i] = VitsWeights.Bias(w, $"{prefix}.res_skip_layers.{i}");
        }
        if (w.ContainsKey($"{prefix}.cond_layer.weight") || w.ContainsKey($"{prefix}.cond_layer.weight_g"))
        {
            _condW = VitsWeights.Conv(w, $"{prefix}.cond_layer");
            _condB = VitsWeights.Bias(w, $"{prefix}.cond_layer");
        }
    }

    /// <summary>Runs the stack over <paramref name="x"/> <c>[1, hidden, T]</c>; <paramref name="g"/> is the optional speaker embedding <c>[1, gin, 1]</c> (multispeaker), projected once and sliced per layer into the gate.</summary>
    /// <returns>The accumulated skip sum.</returns>
    public Tensor Forward(IBackend backend, Tensor x, int t, Tensor? g = null)
    {
        int h = _hidden;
        Tensor output = new(new TensorShape(1, h, t), DType.F32);
        Zero(output);
        Tensor cur = Clone(x);

        // Project g once → [1, 2*h*layers, 1]; sliced per layer below.
        Tensor? gCond = null;
        if (g is not null && _condW is not null)
        {
            gCond = new Tensor(new TensorShape(1, 2 * h * _layers, 1), DType.F32);
            backend.Conv1d(gCond, g, _condW, _condB, 1, 0, 0, 1, 1);
        }

        for (int i = 0; i < _layers; i++)
        {
            int dilation = Pow(_dilationRate, i);
            int pad = dilation * (_kernel - 1) / 2;
            Tensor xIn = new(new TensorShape(1, 2 * h, t), DType.F32);
            backend.Conv1d(xIn, cur, _inW[i]!, _inB[i], 1, pad, pad, dilation, 1);

            // Add the per-layer speaker-conditioning slice (broadcast over time).
            if (gCond is not null)
            {
                float* xip = (float*)xIn.DataPointer;
                float* gp = (float*)gCond.DataPointer + (long)i * 2 * h;
                for (int c = 0; c < 2 * h; c++)
                    for (int j = 0; j < t; j++) xip[(long)c * t + j] += gp[c];
            }

            // fused tanh(first h) * sigmoid(second h) → acts [1, h, t].
            Tensor acts = new(new TensorShape(1, h, t), DType.F32);
            float* xp = (float*)xIn.DataPointer;
            float* ap = (float*)acts.DataPointer;
            for (int c = 0; c < h; c++)
                for (int j = 0; j < t; j++)
                {
                    float a = xp[(long)c * t + j];
                    float b = xp[(long)(h + c) * t + j];
                    ap[(long)c * t + j] = MathF.Tanh(a) * (1f / (1f + MathF.Exp(-b)));
                }
            xIn.Dispose();

            int rsCh = i < _layers - 1 ? 2 * h : h;
            Tensor rs = new(new TensorShape(1, rsCh, t), DType.F32);
            backend.Conv1d(rs, acts, _rsW[i]!, _rsB[i], 1, 0, 0, 1, 1);
            acts.Dispose();

            float* rp = (float*)rs.DataPointer;
            float* op = (float*)output.DataPointer;
            if (i < _layers - 1)
            {
                // residual = (cur + rs[:h]); skip = rs[h:].
                float* cp = (float*)cur.DataPointer;
                for (int c = 0; c < h; c++)
                    for (int j = 0; j < t; j++)
                    {
                        cp[(long)c * t + j] += rp[(long)c * t + j];
                        op[(long)c * t + j] += rp[(long)(h + c) * t + j];
                    }
            }
            else
            {
                for (long n = 0; n < (long)h * t; n++) op[n] += rp[n];
            }
            rs.Dispose();
        }
        cur.Dispose();
        gCond?.Dispose();
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[][] groups = [_inW, _inB, _rsW, _rsB];
        foreach (Tensor?[] g in groups) foreach (Tensor? t in g) if (t is not null) yield return t;
        if (_condW is not null) yield return _condW;
        if (_condB is not null) yield return _condB;
    }

    private static int Pow(int b, int e) { int r = 1; for (int i = 0; i < e; i++) r *= b; return r; }
    private static void Zero(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = 0f; }
    private static Tensor Clone(Tensor t)
    {
        Tensor c = new(t.Shape, DType.F32);
        Buffer.MemoryCopy((void*)t.DataPointer, (void*)c.DataPointer, t.ElementCount * 4, t.ElementCount * 4);
        return c;
    }
}
