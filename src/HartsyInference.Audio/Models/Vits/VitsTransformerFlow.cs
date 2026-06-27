using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Vits;

/// <summary>The Bert-VITS2 <c>TransformerCouplingBlock</c> flow (used by MeloTTS-English-v3): N mean-only
/// <c>TransformerCouplingLayer</c>s, each interleaved with a channel <c>Flip</c>. Every coupling splits the 192
/// latent channels in half, runs a <see cref="VitsFftBlock"/> (3-layer FFT, FFN kernel 5, speaker-conditioned) over
/// <c>pre(x0)</c>, and shifts <c>x1</c> by the projected output (no scale). Inference runs the stack in reverse.
/// Channels-first <c>[1, 192, T]</c>.</summary>
public sealed unsafe class VitsTransformerFlow
{
    private readonly int _channels, _half;
    private readonly Coupling[] _flows;

    public VitsTransformerFlow(VitsConfig cfg, int nFlows = 4, int encLayers = 3, int ffnKernel = 5)
    {
        _channels = cfg.InterChannels;
        _half = _channels / 2;
        _flows = new Coupling[nFlows];
        for (int i = 0; i < nFlows; i++)
            _flows[i] = new Coupling(_channels, _half, cfg.HiddenChannels, cfg.NumHeads, cfg.FilterChannels, encLayers, ffnKernel, cfg.WindowSize);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "flow")
    {
        // flows live at even indices (0,2,4,6); the odd indices are parameterless Flips.
        for (int i = 0; i < _flows.Length; i++) _flows[i].LoadWeights(w, $"{prefix}.flows.{2 * i}");
    }

    /// <summary>Reverse pass: z_p → z. Iterates the [coupling, flip] stack backwards, each coupling in reverse mode
    /// (<c>x1 -= m</c>). Returns a new tensor; <paramref name="zP"/> is consumed.</summary>
    public Tensor Reverse(IBackend backend, Tensor zP, int t, Tensor? g = null)
    {
        Tensor x = zP;
        for (int i = _flows.Length - 1; i >= 0; i--)
        {
            FlipChannelsInPlace(x, t);          // the Flip that follows coupling i (run first in reverse)
            _flows[i].ReverseInPlace(backend, x, t, g);
        }
        return x;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Coupling c in _flows) foreach (Tensor t in c.EnumerateWeights()) yield return t;
    }

    // Flip reverses the full channel order (torch.flip(x, [1])).
    private void FlipChannelsInPlace(Tensor x, int t)
    {
        float* xp = (float*)x.DataPointer;
        for (int c = 0; c < _channels / 2; c++)
        {
            int d = _channels - 1 - c;
            for (int j = 0; j < t; j++)
            {
                float tmp = xp[(long)c * t + j];
                xp[(long)c * t + j] = xp[(long)d * t + j];
                xp[(long)d * t + j] = tmp;
            }
        }
    }

    private sealed class Coupling
    {
        private readonly int _channels, _half;
        private readonly VitsFftBlock _enc;
        private Tensor? _preW, _preB, _postW, _postB;

        public Coupling(int channels, int half, int hidden, int heads, int filter, int encLayers, int ffnKernel, int window)
        {
            _channels = channels; _half = half;
            _enc = new VitsFftBlock(encLayers, hidden, heads, filter, ffnKernel, window);
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
        {
            _preW = VitsWeights.Conv(w, $"{prefix}.pre"); _preB = VitsWeights.Bias(w, $"{prefix}.pre");
            _postW = VitsWeights.Conv(w, $"{prefix}.post"); _postB = VitsWeights.Bias(w, $"{prefix}.post");
            _enc.LoadWeights(w, $"{prefix}.enc");
        }

        // x1 = x1 - m, where m = post(enc(pre(x0), g)). Mean-only (no scale).
        public void ReverseInPlace(IBackend backend, Tensor x, int t, Tensor? g)
        {
            float* xp = (float*)x.DataPointer;
            // h = pre(x0): x0 = first half channels.
            Tensor x0 = new(new TensorShape(1, _half, t), DType.F32);
            float* x0p = (float*)x0.DataPointer;
            for (int c = 0; c < _half; c++)
                for (int j = 0; j < t; j++) x0p[(long)c * t + j] = xp[(long)c * t + j];
            // pre projects half -> hidden (192).
            Tensor hh = new(new TensorShape(1, 192, t), DType.F32);
            backend.Conv1d(hh, x0, _preW!, _preB, 1, 0, 0, 1, 1);
            x0.Dispose();
            hh = _enc.Run(backend, hh, t, g, condLayerIdx: 2);
            // m = post(enc) -> half channels.
            Tensor m = new(new TensorShape(1, _half, t), DType.F32);
            backend.Conv1d(m, hh, _postW!, _postB, 1, 0, 0, 1, 1);
            hh.Dispose();
            float* mp = (float*)m.DataPointer;
            for (int c = 0; c < _half; c++)
                for (int j = 0; j < t; j++) xp[(long)(_half + c) * t + j] -= mp[(long)c * t + j];   // x1 -= m
            m.Dispose();
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            if (_preW is not null) yield return _preW;
            if (_preB is not null) yield return _preB;
            if (_postW is not null) yield return _postW;
            if (_postB is not null) yield return _postB;
            foreach (Tensor t in _enc.EnumerateWeights()) yield return t;
        }
    }
}
