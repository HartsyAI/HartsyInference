using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Demucs;

/// <summary>HTDemucs DConv residual branch: a stack of <c>depth</c> dilated 1×3 conv bottlenecks added back to the
/// input with a per-channel LayerScale (init 1e-4). Each bottleneck is
/// <c>Conv1d(C→hidden, k3, dilation) → GroupNorm(1) → GELU → Conv1d(hidden→2C, k1) → GroupNorm(1) → GLU → LayerScale</c>,
/// dilation doubling per layer (1, 2, …). The released depth-4 main convs keep this branch on for the shallow
/// encoder/decoder layers; the LSTM and local-attention variants are OFF in HTDemucs. Reuses the backend
/// <c>Conv1d</c>/<c>GroupNorm</c>/<c>Gelu</c>.</summary>
public sealed unsafe class DemucsDConv : IDisposable
{
    private readonly DemucsCastOwner _casts = new();
    private readonly int _channels, _depth, _hidden;
    private readonly bool _norm;
    private readonly Layer[] _layers;

    public DemucsDConv(int channels, int depth = 2, int compress = 4, bool norm = true)
    {
        _channels = channels; _depth = depth; _norm = norm;
        _hidden = channels / compress;
        _layers = new Layer[depth];
        for (int i = 0; i < depth; i++) _layers[i] = new Layer(channels, _hidden, 1 << i, norm);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        for (int i = 0; i < _depth; i++) _layers[i].LoadWeights(w, $"{prefix}.layers.{i}", _casts);
    }

    /// <summary>Adds the residual DConv branch in place onto <paramref name="x"/> (<c>[1, C, T]</c>).</summary>
    public void Forward(IBackend backend, Tensor x, int t)
    {
        for (int i = 0; i < _depth; i++) _layers[i].Forward(backend, x, t);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int i = 0; i < _depth; i++) foreach (Tensor w in _layers[i].EnumerateWeights()) yield return w;
    }

    /// <summary>Frees the F32 casts this DConv stack allocated at load time.</summary>
    public void Dispose() => _casts.Dispose();

    private sealed class Layer
    {
        private readonly int _channels, _dilation;
        private int _hidden;       // inferred from the conv1 weight at load (htdemucs DConv uses compress=8)
        private readonly bool _norm;
        private Tensor? _conv1W, _conv1B, _conv2W, _conv2B, _n1W, _n1B, _n2W, _n2B, _scale;

        public Layer(int channels, int hidden, int dilation, bool norm)
        {
            _channels = channels; _hidden = hidden; _dilation = dilation; _norm = norm;
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p, DemucsCastOwner casts)
        {
            // Demucs DConv Sequential indices: 0=Conv1d, (1=GroupNorm), GELU, Conv1d, (GroupNorm), GLU, LayerScale.
            // With norm on: 0 conv, 1 norm, 2 gelu(no params), 3 conv, 4 norm, 5 glu, 6 layerscale.
            // With norm off: 0 conv, 1 gelu, 2 conv, 3 glu, 4 layerscale.
            int c1 = 0, c2, ls;
            if (_norm)
            {
                _n1W = casts.F32(w, $"{p}.1.weight"); _n1B = casts.F32(w, $"{p}.1.bias");
                c2 = 3;
                _n2W = casts.F32(w, $"{p}.4.weight"); _n2B = casts.F32(w, $"{p}.4.bias");
                ls = 6;
            }
            else { c2 = 2; ls = 4; }
            _conv1W = casts.F32(w, $"{p}.{c1}.weight"); _conv1B = casts.Optional(w, $"{p}.{c1}.bias");
            _hidden = (int)_conv1W.Shape[0];      // htdemucs uses compress=8 (hidden = channels/8)
            _conv2W = casts.F32(w, $"{p}.{c2}.weight"); _conv2B = casts.Optional(w, $"{p}.{c2}.bias");
            _scale = casts.F32(w, $"{p}.{ls}.scale");
        }

        public void Forward(IBackend backend, Tensor x, int t)
        {
            Tensor h = new(new TensorShape(1, _hidden, t), DType.F32);
            backend.Conv1d(h, x, _conv1W!, _conv1B, 1, _dilation, _dilation, _dilation, 1);
            // GroupNorm(1, C) over (C, T) — done inline because the backend GroupNorm needs a rank-4 [N,C,H,W].
            if (_norm) GroupNorm1(h, _hidden, t, _n1W!, _n1B!);
            HartsyInference.Audio.Layers.Activations.ErfGelu(h); Tensor g = h;   // demucs F.gelu = exact erf
            Tensor c2 = new(new TensorShape(1, 2 * _channels, t), DType.F32);
            backend.Conv1d(c2, g, _conv2W!, _conv2B, 1, 0, 0, 1, 1); g.Dispose();
            if (_norm) GroupNorm1(c2, 2 * _channels, t, _n2W!, _n2B!);
            // GLU(dim=1): split channels → a · σ(b), then LayerScale, then residual add in place.
            float* gp = (float*)c2.DataPointer;
            float* xp = (float*)x.DataPointer;
            float* sp = (float*)_scale!.DataPointer;
            for (int ch = 0; ch < _channels; ch++)
            {
                float s = sp[ch];
                for (int j = 0; j < t; j++)
                {
                    float a = gp[(long)ch * t + j];
                    float b = gp[(long)(_channels + ch) * t + j];
                    xp[(long)ch * t + j] += s * (a * (1f / (1f + MathF.Exp(-b))));
                }
            }
            c2.Dispose();
        }

        /// <summary>GroupNorm(num_groups=1) over a channels-first [1, C, T] tensor in place: normalize over all
        /// C·T (biased variance, eps 1e-5), then per-channel affine. Matches torch nn.GroupNorm(1, C).</summary>
        private static void GroupNorm1(Tensor x, int c, int t, Tensor weight, Tensor bias)
        {
            float* xp = (float*)x.DataPointer; float* wp = (float*)weight.DataPointer; float* bp = (float*)bias.DataPointer;
            long n = (long)c * t;
            double sum = 0, sumSq = 0;
            for (long i = 0; i < n; i++) { float v = xp[i]; sum += v; sumSq += (double)v * v; }
            double mean = sum / n;
            double var = sumSq / n - mean * mean;
            float invStd = (float)(1.0 / Math.Sqrt(var + 1e-5));
            for (int ch = 0; ch < c; ch++)
            {
                float wc = wp[ch], bc = bp[ch];
                float* row = xp + (long)ch * t;
                for (int j = 0; j < t; j++) row[j] = (float)((row[j] - mean) * invStd) * wc + bc;
            }
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] all = [_conv1W, _conv1B, _conv2W, _conv2B, _n1W, _n1B, _n2W, _n2B, _scale];
            foreach (Tensor? w in all) if (w is not null) yield return w;
        }
    }
}
