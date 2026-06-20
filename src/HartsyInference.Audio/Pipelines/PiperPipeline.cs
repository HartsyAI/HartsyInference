using HartsyInference.Audio.Models.Vits;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Piper TTS pipeline — a thin wrapper over the shared <see cref="VitsSynthesizer"/> with Piper's
/// medium config and phoneme interspersing (BOS=1, blank=0 between phonemes, EOS=2). Takes espeak phoneme
/// ids (caller phonemizes via espeak-ng + the voice's <c>phoneme_id_map</c>); the Audio package carries no
/// phonemizer dependency.</summary>
public sealed class PiperPipeline : IDisposable
{
    private readonly VitsConfig _cfg;
    private readonly VitsSynthesizer _synth;
    private int _disposed;

    public PiperPipeline(VitsConfig cfg)
    {
        _cfg = cfg;
        _synth = new VitsSynthesizer(cfg);
    }

    public int SampleRate => _cfg.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w) => _synth.LoadWeights(w);

    /// <summary>Synthesizes audio from espeak phoneme ids. Inserts the VITS blank between phonemes and wraps
    /// with BOS/EOS before running the synthesizer.</summary>
    public float[] Synthesize(IBackend backend, ReadOnlySpan<int> phonemeIds, float? lengthScale = null,
        float? noiseScale = null, int seed = 0)
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(PiperPipeline));
        int[] tokens = VitsLengthRegulator.Intersperse(phonemeIds, blank: 0, bos: 1, eos: 2);
        return _synth.Infer(backend, tokens, lengthScale ?? _cfg.LengthScale, noiseScale ?? _cfg.NoiseScale, seed);
    }

    public IEnumerable<Tensor> EnumerateWeights() => _synth.EnumerateWeights();

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _synth.Dispose();
        GC.SuppressFinalize(this);
    }
}
