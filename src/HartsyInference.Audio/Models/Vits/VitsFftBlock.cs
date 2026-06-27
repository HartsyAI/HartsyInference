using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Vits;

/// <summary>A stack of VITS/Bert-VITS2 FFT blocks (the <c>attentions.Encoder</c>): pre-norm relative-position
/// multi-head attention + Conv FFN, with optional speaker conditioning (<c>x += spk_emb_linear(g)</c> before a
/// chosen layer). Shared by the text encoder (6 layers, FFN kernel 3) and the transformer-coupling flow
/// (3 layers, FFN kernel 5). Channels-first <c>[1, hidden, T]</c>. The attention math is validated bit-exact
/// against the reference text encoder.</summary>
public sealed unsafe class VitsFftBlock
{
    private readonly int _hidden, _heads, _kCh, _filter, _ffnKernel, _window;
    private readonly Layer[] _layers;
    private Tensor? _spkW, _spkB;

    public VitsFftBlock(int numLayers, int hidden, int heads, int filterChannels, int ffnKernel, int window)
    {
        _hidden = hidden; _heads = heads; _kCh = hidden / heads; _filter = filterChannels;
        _ffnKernel = ffnKernel; _window = window;
        _layers = new Layer[numLayers];
        for (int i = 0; i < numLayers; i++) _layers[i] = new Layer(hidden, heads, _kCh, filterChannels, ffnKernel, window);
    }

    /// <summary>Loads the per-layer attention/FFN/norm weights under <paramref name="prefix"/> (the
    /// <c>...encoder</c> module), and the optional <c>spk_emb_linear</c> speaker projection.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, prefix, i);
        if (w.TryGetValue($"{prefix}.spk_emb_linear.weight", out Tensor? sw))
        { _spkW = sw; _spkB = w.TryGetValue($"{prefix}.spk_emb_linear.bias", out Tensor? sb) ? sb : null; }
    }

    /// <summary>Runs the FFT layers over <paramref name="x"/> <c>[1, hidden, T]</c> (takes ownership, returns a new
    /// tensor). When <paramref name="g"/> <c>[1, gin, 1]</c> and <c>spk_emb_linear</c> are present, adds the projected
    /// speaker vector before layer <paramref name="condLayerIdx"/>.</summary>
    public Tensor Run(IBackend backend, Tensor x, int t, Tensor? g = null, int condLayerIdx = 2)
    {
        float[]? gProj = null;
        if (g is not null && _spkW is not null)
        {
            gProj = new float[_hidden];
            float* gp = (float*)g.DataPointer, sw = (float*)_spkW.DataPointer, sb = _spkB is null ? null : (float*)_spkB.DataPointer;
            int gin = (int)(_spkW.Shape.ElementCount / _hidden);
            for (int c = 0; c < _hidden; c++)
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
                for (int c = 0; c < _hidden; c++)
                    for (int j = 0; j < t; j++) xp[(long)c * t + j] += gProj[c];
            }
            Tensor next = _layers[i].Forward(backend, x, t);
            x.Dispose(); x = next;
        }
        return x;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Layer l in _layers) foreach (Tensor t in l.EnumerateWeights()) yield return t;
        if (_spkW is not null) yield return _spkW;
        if (_spkB is not null) yield return _spkB;
    }

    private sealed class Layer
    {
        private readonly int _h, _heads, _kCh, _filter, _ffnKernel, _w;
        private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _relK, _relV;
        private Tensor? _norm1G, _norm1B, _norm2G, _norm2B, _ffn1W, _ffn1B, _ffn2W, _ffn2B;

        public Layer(int hidden, int heads, int kCh, int filter, int ffnKernel, int window)
        { _h = hidden; _heads = heads; _kCh = kCh; _filter = filter; _ffnKernel = ffnKernel; _w = window; }

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
            int pad = (_ffnKernel - 1) / 2;
            Tensor attn = Attention(backend, x, t);
            Tensor afterAttn = AddNorm(x, attn, _norm1G!, _norm1B!, _h, t); attn.Dispose();

            Tensor f1 = new(new TensorShape(1, _filter, t), DType.F32);
            backend.Conv1d(f1, afterAttn, _ffn1W!, _ffn1B, 1, pad, pad, 1, 1);
            float* fp = (float*)f1.DataPointer;
            for (long n = 0; n < f1.ElementCount; n++) if (fp[n] < 0) fp[n] = 0;     // ReLU
            Tensor f2 = new(new TensorShape(1, _h, t), DType.F32);
            backend.Conv1d(f2, f1, _ffn2W!, _ffn2B, 1, pad, pad, 1, 1); f1.Dispose();
            Tensor outT = AddNorm(afterAttn, f2, _norm2G!, _norm2B!, _h, t);
            afterAttn.Dispose(); f2.Dispose();
            return outT;
        }

        private Tensor Attention(IBackend backend, Tensor x, int t)
        {
            int h = _h, heads = _heads, kc = _kCh, w = _w;
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
                        int diff = j - i;
                        bool inWindow = diff >= -w && diff <= w;
                        int rel = diff + w;
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
                            float relVv = inWindow ? rv[(long)(diff + w) * kc + c] : 0f;
                            acc += p * (vp[(long)(chBase + c) * t + j] + relVv);
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
