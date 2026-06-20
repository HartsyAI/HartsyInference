using HartsyInference.Audio.Dsp;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Vits;

/// <summary>VITS posterior encoder (<c>enc_q</c>): <c>pre(Conv1d) → WaveNet(g) → proj</c> → (m, logs); samples
/// <c>z = m + randn·exp(logs)</c>. Training-only for plain TTS, but the inference path for voice conversion
/// (OpenVoice / GPT-SoVITS). Reuses <see cref="VitsWaveNet"/>. Channels-first <c>[1, spec, T]</c> in.</summary>
public sealed unsafe class VitsPosteriorEncoder
{
    private readonly int _spec, _hidden, _inter;
    private readonly VitsWaveNet _wn;
    private Tensor? _preW, _preB, _projW, _projB;

    public VitsPosteriorEncoder(int specChannels, int hidden, int inter, int kernel, int dilationRate, int layers)
    {
        _spec = specChannels; _hidden = hidden; _inter = inter;
        _wn = new VitsWaveNet(hidden, kernel, dilationRate, layers);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "enc_q")
    {
        _preW = VitsWeights.Conv(w, $"{prefix}.pre"); _preB = VitsWeights.Bias(w, $"{prefix}.pre");
        _wn.LoadWeights(w, $"{prefix}.enc");
        _projW = VitsWeights.Conv(w, $"{prefix}.proj"); _projB = VitsWeights.Bias(w, $"{prefix}.proj");
    }

    /// <summary>Encodes a spectrogram <c>[1, spec, T]</c> → sampled latent <c>z [1, inter, T]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor spec, int t, Tensor? g, int seed = 0)
    {
        Tensor h = new(new TensorShape(1, _hidden, t), DType.F32);
        backend.Conv1d(h, spec, _preW!, _preB, 1, 0, 0, 1, 1);
        Tensor hOut = _wn.Forward(backend, h, t, g); h.Dispose();
        Tensor stats = new(new TensorShape(1, 2 * _inter, t), DType.F32);
        backend.Conv1d(stats, hOut, _projW!, _projB, 1, 0, 0, 1, 1); hOut.Dispose();

        Tensor z = new(new TensorShape(1, _inter, t), DType.F32);
        float* sp = (float*)stats.DataPointer;
        float* zp = (float*)z.DataPointer;
        uint rng = DeterministicRng.Seed(seed);
        for (int c = 0; c < _inter; c++)
            for (int j = 0; j < t; j++)
            {
                float m = sp[(long)c * t + j];
                float logs = sp[(long)(_inter + c) * t + j];
                zp[(long)c * t + j] = m + DeterministicRng.NextGaussian(ref rng) * MathF.Exp(logs);
            }
        stats.Dispose();
        return z;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_preW is not null) yield return _preW;
        if (_preB is not null) yield return _preB;
        foreach (Tensor t in _wn.EnumerateWeights()) yield return t;
        if (_projW is not null) yield return _projW;
        if (_projB is not null) yield return _projB;
    }
}
