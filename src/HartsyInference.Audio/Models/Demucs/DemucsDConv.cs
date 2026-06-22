using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Demucs;

/// <summary>HTDemucs DConv residual branch: a stack of <c>depth</c> dilated 1×3 conv bottlenecks added back to the
/// input with a per-channel LayerScale (init 1e-4). Each bottleneck is
/// <c>Conv1d(C→hidden, k3, dilation) → GroupNorm(1) → GELU → Conv1d(hidden→2C, k1) → GroupNorm(1) → GLU → LayerScale</c>,
/// dilation doubling per layer (1, 2, …). The released depth-4 main convs keep this branch on for the shallow
/// encoder/decoder layers; the LSTM and local-attention variants are OFF in HTDemucs. Reuses the backend
/// <c>Conv1d</c>/<c>GroupNorm</c>/<c>Gelu</c>.</summary>
public sealed unsafe class DemucsDConv
{
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
        for (int i = 0; i < _depth; i++) _layers[i].LoadWeights(w, $"{prefix}.layers.{i}");
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

    private sealed class Layer
    {
        private readonly int _channels, _hidden, _dilation;
        private readonly bool _norm;
        private Tensor? _conv1W, _conv1B, _conv2W, _conv2B, _n1W, _n1B, _n2W, _n2B, _scale;

        public Layer(int channels, int hidden, int dilation, bool norm)
        {
            _channels = channels; _hidden = hidden; _dilation = dilation; _norm = norm;
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            // Demucs DConv Sequential indices: 0=Conv1d, (1=GroupNorm), GELU, Conv1d, (GroupNorm), GLU, LayerScale.
            // With norm on: 0 conv, 1 norm, 2 gelu(no params), 3 conv, 4 norm, 5 glu, 6 layerscale.
            // With norm off: 0 conv, 1 gelu, 2 conv, 3 glu, 4 layerscale.
            int c1 = 0, c2, ls;
            if (_norm)
            {
                _n1W = WhisperOps.EnsureF32(w[$"{p}.1.weight"]); _n1B = WhisperOps.EnsureF32(w[$"{p}.1.bias"]);
                c2 = 3;
                _n2W = WhisperOps.EnsureF32(w[$"{p}.4.weight"]); _n2B = WhisperOps.EnsureF32(w[$"{p}.4.bias"]);
                ls = 6;
            }
            else { c2 = 2; ls = 4; }
            _conv1W = WhisperOps.EnsureF32(w[$"{p}.{c1}.weight"]); _conv1B = Bias(w, $"{p}.{c1}.bias");
            _conv2W = WhisperOps.EnsureF32(w[$"{p}.{c2}.weight"]); _conv2B = Bias(w, $"{p}.{c2}.bias");
            _scale = WhisperOps.EnsureF32(w[$"{p}.{ls}.scale"]);
        }

        public void Forward(IBackend backend, Tensor x, int t)
        {
            Tensor h = new(new TensorShape(1, _hidden, t), DType.F32);
            backend.Conv1d(h, x, _conv1W!, _conv1B, 1, _dilation, _dilation, _dilation, 1);
            if (_norm)
            {
                Tensor hn = new(h.Shape, DType.F32); backend.GroupNorm(hn, h, _n1W!, _n1B!, 1, 1e-5f); h.Dispose(); h = hn;
            }
            Tensor g = new(h.Shape, DType.F32); backend.Gelu(g, h); h.Dispose();
            Tensor c2 = new(new TensorShape(1, 2 * _channels, t), DType.F32);
            backend.Conv1d(c2, g, _conv2W!, _conv2B, 1, 0, 0, 1, 1); g.Dispose();
            if (_norm)
            {
                Tensor cn = new(c2.Shape, DType.F32); backend.GroupNorm(cn, c2, _n2W!, _n2B!, 1, 1e-5f); c2.Dispose(); c2 = cn;
            }
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

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] all = [_conv1W, _conv1B, _conv2W, _conv2B, _n1W, _n1B, _n2W, _n2B, _scale];
            foreach (Tensor? w in all) if (w is not null) yield return w;
        }

        private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key)
            => w.TryGetValue(key, out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
    }
}
