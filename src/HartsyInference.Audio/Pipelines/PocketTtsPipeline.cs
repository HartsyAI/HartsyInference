using System.Diagnostics;
using HartsyInference.Audio.Models.PocketTts;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Kyutai Pocket-TTS pipeline: wires tokenized text + a voice (a built-in voice name → its stored
/// continuous-latent priming state, or a raw audio prompt → <see cref="MimiContinuousLatent.EncodeToLatent"/>
/// priming for zero-shot cloning) through the continuous-latent AR <see cref="PocketTtsModel"/> to 24 kHz mono PCM.
///
/// <para>The Audio package carries no text-BPE dependency, so the caller supplies pre-tokenized text (the
/// SentencePiece <c>tokenizer.model</c> shipped with the checkpoint is applied upstream). Built-in voice priming
/// states are registered via <see cref="RegisterVoice"/> — load them from the checkpoint's
/// <c>embeddings*/</c> / <c>languages/</c> voice dirs as <c>[1, LatentDim, T]</c> latent tensors.</para></summary>
public sealed unsafe class PocketTtsPipeline : IDisposable
{
    private readonly PocketTtsConfig _cfg;
    private readonly PocketTtsModel _model;
    private readonly Dictionary<string, Tensor> _voiceStates = new(StringComparer.OrdinalIgnoreCase);
    private int _disposed;

    public PocketTtsConfig Config => _cfg;
    public int SampleRate => _cfg.SampleRate;

    public PocketTtsPipeline(PocketTtsConfig cfg)
    {
        _cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
        _model = new PocketTtsModel(cfg);
    }

    /// <summary>Loads the Flow-LM transformer, the per-frame flow head, and the continuous-Mimi codec. See
    /// <see cref="PocketTtsModel.LoadWeights"/> for the prefix conventions.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w,
        string lmPrefix = "flow_lm.transformer", string flowHeadPrefix = "flow_lm.flow_head")
    {
        ThrowIfDisposed();
        _model.LoadWeights(w, lmPrefix, flowHeadPrefix);
    }

    /// <summary>Registers a built-in voice's priming state — a continuous latent <c>[1, LatentDim, T]</c> that the
    /// AR loop is primed with (the codec-encoded reference, stored in the checkpoint's voice dirs). The pipeline
    /// takes ownership and disposes it on <see cref="Dispose"/>.</summary>
    public void RegisterVoice(string name, Tensor primingLatents)
    {
        ThrowIfDisposed();
        if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("voice name must be non-empty.", nameof(name));
        if (primingLatents.Shape.Rank != 3 || (int)primingLatents.Shape[1] != _cfg.LatentDim)
            throw new ArgumentException($"voice priming latents must be [1, {_cfg.LatentDim}, T], got {primingLatents.Shape}.", nameof(primingLatents));
        if (_voiceStates.Remove(name, out Tensor? old)) old.Dispose();
        _voiceStates[name] = primingLatents;
    }

    /// <summary>Encodes a raw 24 kHz mono audio prompt to a continuous-latent priming state (zero-shot cloning).
    /// <paramref name="pcm"/> is mono samples; returns <c>[1, LatentDim, T]</c> latents to pass to
    /// <see cref="Synthesize"/>. The caller owns and disposes the result.</summary>
    public Tensor BuildVoiceFromAudio(IBackend backend, ReadOnlySpan<float> pcm)
    {
        ThrowIfDisposed();
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (pcm.Length == 0) throw new ArgumentException("audio prompt is empty.", nameof(pcm));

        Tensor pcmT = new(new TensorShape(1, 1, pcm.Length), DType.F32);
        Span<float> dst = new((void*)pcmT.DataPointer, pcm.Length);
        pcm.CopyTo(dst);
        Tensor latents = _model.Codec.EncodeToLatent(backend, pcmT, 1, pcm.Length);
        pcmT.Dispose();
        return latents;
    }

    /// <summary>Synthesizes 24 kHz mono PCM from pre-tokenized <paramref name="textTokens"/>. Voice selection:
    /// pass a registered built-in <paramref name="voiceName"/>, OR an explicit <paramref name="voiceState"/>
    /// (e.g. from <see cref="BuildVoiceFromAudio"/>); leave both unset for the model's default voice. When both are
    /// given, <paramref name="voiceState"/> wins.</summary>
    public float[] Synthesize(IBackend backend, ReadOnlySpan<int> textTokens, string? voiceName = null,
        Tensor? voiceState = null, int seed = 0, int numFrames = 0)
    {
        ThrowIfDisposed();
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        Stopwatch sw = Stopwatch.StartNew();

        Tensor? prime = voiceState;
        if (prime is null && voiceName is not null)
        {
            if (!_voiceStates.TryGetValue(voiceName, out prime))
                throw new ArgumentException($"unknown voice '{voiceName}'; register it via RegisterVoice or pass voiceState.", nameof(voiceName));
        }

        float[] audio = _model.Generate(backend, textTokens, prime, seed, numFrames);
        sw.Stop();
        Logs.Info($"PocketTTS: {textTokens.Length} text tokens → {audio.Length} samples " +
            $"({audio.Length / (double)SampleRate:F1}s) in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    public IEnumerable<Tensor> EnumerateWeights() => _model.EnumerateWeights();

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _model.Dispose();
        foreach (Tensor t in _voiceStates.Values) t.Dispose();
        _voiceStates.Clear();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(PocketTtsPipeline));
    }
}
