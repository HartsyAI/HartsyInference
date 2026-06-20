using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Vits;

/// <summary>VITS normalizing flow (<c>ResidualCouplingBlock</c>): <c>n_flows</c> mean-only affine coupling
/// layers interleaved with channel flips. At inference it runs in reverse: <c>x1 = x1 − m</c> where <c>m</c>
/// is produced by <c>pre → WaveNet → post</c> over the untouched half <c>x0</c>. Reuses
/// <see cref="VitsWaveNet"/>. Channels-first <c>[1, channels, T]</c>.</summary>
public sealed unsafe class VitsFlow
{
    private readonly int _channels, _half, _nFlows;
    private readonly Coupling[] _layers;

    public VitsFlow(VitsConfig cfg)
    {
        _channels = cfg.InterChannels;
        _half = _channels / 2;
        _nFlows = cfg.FlowFlows;
        _layers = new Coupling[_nFlows];
        for (int i = 0; i < _nFlows; i++)
            _layers[i] = new Coupling(_half, cfg.HiddenChannels, cfg.FlowKernelSize, cfg.FlowDilationRate, cfg.FlowLayers);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "flow")
    {
        // flows = [RCL, Flip, RCL, Flip, ...] → coupling layer i is at flows index 2*i.
        for (int i = 0; i < _nFlows; i++) _layers[i].LoadWeights(w, $"{prefix}.flows.{2 * i}");
    }

    /// <summary>Reverse pass: applies (flip, coupling) for each flow in reverse order. <paramref name="g"/>
    /// is the optional speaker embedding for the WaveNet conditioning.</summary>
    public Tensor Reverse(IBackend backend, Tensor zP, int t, Tensor? g = null)
    {
        Tensor x = Clone(zP);
        for (int i = _nFlows - 1; i >= 0; i--)
        {
            FlipChannels(x, _channels, t);
            Tensor next = _layers[i].Apply(backend, x, t, g, reverse: true);
            x.Dispose();
            x = next;
        }
        return x;
    }

    /// <summary>Forward pass: applies (coupling, flip) for each flow in order — used by the OpenVoice tone
    /// converter to map the posterior into the prior latent under the source speaker.</summary>
    public Tensor Forward(IBackend backend, Tensor z, int t, Tensor? g = null)
    {
        Tensor x = Clone(z);
        for (int i = 0; i < _nFlows; i++)
        {
            Tensor next = _layers[i].Apply(backend, x, t, g, reverse: false);
            x.Dispose();
            x = next;
            FlipChannels(x, _channels, t);
        }
        return x;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Coupling c in _layers) foreach (Tensor t in c.EnumerateWeights()) yield return t;
    }

    private static void FlipChannels(Tensor x, int channels, int t)
    {
        float* p = (float*)x.DataPointer;
        for (int c = 0; c < channels / 2; c++)
        {
            int cc = channels - 1 - c;
            for (int j = 0; j < t; j++)
            {
                long a = (long)c * t + j, b = (long)cc * t + j;
                (p[a], p[b]) = (p[b], p[a]);
            }
        }
    }

    private static Tensor Clone(Tensor t)
    {
        Tensor c = new(t.Shape, DType.F32);
        Buffer.MemoryCopy((void*)t.DataPointer, (void*)c.DataPointer, t.ElementCount * 4, t.ElementCount * 4);
        return c;
    }

    private sealed class Coupling
    {
        private readonly int _half, _hidden;
        private readonly VitsWaveNet _wn;
        private Tensor? _preW, _preB, _postW, _postB;

        public Coupling(int half, int hidden, int kernel, int dilationRate, int layers)
        {
            _half = half; _hidden = hidden;
            _wn = new VitsWaveNet(hidden, kernel, dilationRate, layers);
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
        {
            _preW = VitsWeights.Conv(w, $"{prefix}.pre"); _preB = VitsWeights.Bias(w, $"{prefix}.pre");
            _postW = VitsWeights.Conv(w, $"{prefix}.post"); _postB = VitsWeights.Bias(w, $"{prefix}.post");
            _wn.LoadWeights(w, $"{prefix}.enc");
        }

        public Tensor Apply(IBackend backend, Tensor x, int t, Tensor? g, bool reverse)
        {
            // Split x → x0 (first half, conditioning), x1 (second half, transformed).
            Tensor x0 = SliceChannels(x, 0, _half, t);
            Tensor h = new(new TensorShape(1, _hidden, t), DType.F32);
            backend.Conv1d(h, x0, _preW!, _preB, 1, 0, 0, 1, 1);
            Tensor hOut = _wn.Forward(backend, h, t, g); h.Dispose();
            Tensor m = new(new TensorShape(1, _half, t), DType.F32);
            backend.Conv1d(m, hOut, _postW!, _postB, 1, 0, 0, 1, 1); hOut.Dispose();

            // mean-only (logs=0): reverse x1 = x1 - m; forward x1 = x1 + m. Reassemble [x0, x1].
            Tensor outT = new(x.Shape, DType.F32);
            float* xp = (float*)x.DataPointer;
            float* mp = (float*)m.DataPointer;
            float* op = (float*)outT.DataPointer;
            Buffer.MemoryCopy((void*)x0.DataPointer, op, (long)_half * t * 4, (long)_half * t * 4);
            for (int c = 0; c < _half; c++)
                for (int j = 0; j < t; j++)
                {
                    float x1 = xp[(long)(_half + c) * t + j], mv = mp[(long)c * t + j];
                    op[(long)(_half + c) * t + j] = reverse ? x1 - mv : x1 + mv;
                }
            x0.Dispose(); m.Dispose();
            return outT;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] own = [_preW, _preB, _postW, _postB];
            foreach (Tensor? t in own) if (t is not null) yield return t;
            foreach (Tensor t in _wn.EnumerateWeights()) yield return t;
        }

        private static Tensor SliceChannels(Tensor x, int start, int count, int t)
        {
            Tensor outT = new(new TensorShape(1, count, t), DType.F32);
            Buffer.MemoryCopy((float*)x.DataPointer + (long)start * t, (void*)outT.DataPointer,
                (long)count * t * 4, (long)count * t * 4);
            return outT;
        }
    }
}
