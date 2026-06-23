using HartsyInference.Audio.Models.OpenVoice;
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>OpenVoice v2 Tone Color Converter: a VITS posterior + flow + HiFi-GAN run as a voice converter —
/// encode the source spectrogram, flow it into the prior latent under the <b>source</b> speaker, flow it back
/// out under the <b>target</b> speaker, and decode. Reuses the (now speaker-conditioned) <see cref="VitsFlow"/>,
/// <see cref="VitsPosteriorEncoder"/>, and <see cref="VitsHiFiGan"/>. Stage-1 base TTS is MeloTTS; speaker
/// embeddings are extracted from reference spectrograms by the converter's own
/// <see cref="OpenVoiceSpeakerEncoder"/>.</summary>
public sealed unsafe class OpenVoicePipeline : IDisposable
{
    private readonly VitsConfig _cfg;
    private readonly VitsPosteriorEncoder _enc;
    private readonly VitsFlow _flow;
    private readonly VitsHiFiGan _dec;
    private readonly OpenVoiceSpeakerEncoder _refEnc;
    private int _disposed;

    public OpenVoicePipeline(VitsConfig cfg, int specChannels, int posteriorLayers = 16)
    {
        _cfg = cfg;
        _enc = new VitsPosteriorEncoder(specChannels, cfg.HiddenChannels, cfg.InterChannels,
            cfg.FlowKernelSize, cfg.FlowDilationRate, posteriorLayers);
        _flow = new VitsFlow(cfg);
        _dec = new VitsHiFiGan(cfg);
        _refEnc = new OpenVoiceSpeakerEncoder(new OpenVoiceSpeakerConfig
        {
            SampleRate = cfg.SampleRate,
            SpecChannels = specChannels,
            Gin = cfg.GinChannels,
        });
    }

    public int SampleRate => _cfg.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _enc.LoadWeights(w);
        _flow.LoadWeights(w);
        _dec.LoadWeights(w);
        // The reference (tone-color) encoder is optional: callers that pass precomputed speaker embeddings to
        // Convert(...) don't need it. Load it only when its weights are present; ExtractSpeaker /
        // ConvertWithReferences throw clearly if invoked without it.
        if (w.ContainsKey("ref_enc.layernorm.weight")) _refEnc.LoadWeights(w);
    }

    /// <summary>Extracts the tone-color embedding <c>g [1, gin, 1]</c> from a reference linear spectrogram
    /// <c>[1, spec, t]</c> (the converter's <c>extract_se</c>). Caller owns the returned tensor.</summary>
    public Tensor ExtractSpeaker(IBackend backend, Tensor spec, int t)
    {
        ThrowIfDisposed();
        return _refEnc.Extract(backend, spec, t);
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

    /// <summary>End-to-end convert: extracts source + target tone-color from their reference spectrograms, then
    /// runs <see cref="Convert(IBackend, Tensor, int, Tensor, Tensor, int)"/> on the source spectrogram.</summary>
    public float[] ConvertWithReferences(IBackend backend, Tensor srcSpec, int srcT, Tensor tgtSpec, int tgtT, int seed = 0)
    {
        ThrowIfDisposed();
        Tensor gSrc = _refEnc.Extract(backend, srcSpec, srcT);
        Tensor gTgt = _refEnc.Extract(backend, tgtSpec, tgtT);
        try
        {
            return Convert(backend, srcSpec, srcT, gSrc, gTgt, seed);
        }
        finally
        {
            gSrc.Dispose();
            gTgt.Dispose();
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _enc.EnumerateWeights()) yield return t;
        foreach (Tensor t in _flow.EnumerateWeights()) yield return t;
        foreach (Tensor t in _dec.EnumerateWeights()) yield return t;
        foreach (Tensor t in _refEnc.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _refEnc.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(OpenVoicePipeline));
    }
}
