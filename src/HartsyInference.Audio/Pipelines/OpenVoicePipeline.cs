using HartsyInference.Audio.Models.Vits;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>OpenVoice v2 Tone Color Converter: a VITS posterior + flow + HiFi-GAN run as a voice converter —
/// encode the source spectrogram, flow it into the prior latent under the <b>source</b> speaker, flow it back
/// out under the <b>target</b> speaker, and decode. Reuses the (now speaker-conditioned) <see cref="VitsFlow"/>,
/// <see cref="VitsPosteriorEncoder"/>, and <see cref="VitsHiFiGan"/>. Stage-1 base TTS is MeloTTS; the speaker
/// embeddings are caller-supplied (the reference encoder is staged).</summary>
public sealed unsafe class OpenVoicePipeline : IDisposable
{
    private readonly VitsConfig _cfg;
    private readonly VitsPosteriorEncoder _enc;
    private readonly VitsFlow _flow;
    private readonly VitsHiFiGan _dec;
    private int _disposed;

    public OpenVoicePipeline(VitsConfig cfg, int specChannels, int posteriorLayers = 16)
    {
        _cfg = cfg;
        _enc = new VitsPosteriorEncoder(specChannels, cfg.HiddenChannels, cfg.InterChannels,
            cfg.FlowKernelSize, cfg.FlowDilationRate, posteriorLayers);
        _flow = new VitsFlow(cfg);
        _dec = new VitsHiFiGan(cfg);
    }

    public int SampleRate => _cfg.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _enc.LoadWeights(w);
        _flow.LoadWeights(w);
        _dec.LoadWeights(w);
    }

    /// <summary>Converts a source spectrogram's tone color from <paramref name="gSrc"/> to <paramref name="gTgt"/>
    /// (both <c>[1, gin, 1]</c> speaker embeddings) → 24 kHz PCM. <paramref name="spec"/> is <c>[1, spec, T]</c>.</summary>
    public float[] Convert(IBackend backend, Tensor spec, int t, Tensor gSrc, Tensor gTgt, int seed = 0)
    {
        ThrowIfDisposed();
        Tensor z = _enc.Forward(backend, spec, t, gSrc, seed);
        Tensor zP = _flow.Forward(backend, z, t, gSrc); z.Dispose();
        Tensor zHat = _flow.Reverse(backend, zP, t, gTgt); zP.Dispose();
        float[] audio = _dec.Forward(backend, zHat, t, gTgt); zHat.Dispose();
        return audio;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _enc.EnumerateWeights()) yield return t;
        foreach (Tensor t in _flow.EnumerateWeights()) yield return t;
        foreach (Tensor t in _dec.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(OpenVoicePipeline));
    }
}
