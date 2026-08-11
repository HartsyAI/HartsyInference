using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VibeVoice;

/// <summary>VibeVoice-Realtime's stop signal: a 2-layer MLP (<c>Linear(hidden→hidden) → ReLU →
/// Linear(hidden→1)</c>) on the TTS backbone's last hidden state, thresholded after a sigmoid. Replaces the
/// multi-speaker variant's token-based EOS (there is no <c>lm_head</c> on this variant at all — real weight
/// keys confirm the checkpoint has no vocabulary projection for the TTS backbone). Top-level checkpoint keys
/// (<c>tts_eos_classifier.fc1.*</c> / <c>fc2.*</c> — not under the <c>model.</c> prefix everything else uses).</summary>
internal sealed unsafe class VibeVoiceEosClassifier
{
    private readonly int _hidden;
    private Tensor? _fc1W, _fc1B, _fc2W, _fc2B;

    public VibeVoiceEosClassifier(int hidden)
    {
        if (hidden <= 0) throw new ArgumentException("hidden must be positive.", nameof(hidden));
        _hidden = hidden;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _fc1W = WhisperOps.EnsureF32(w["tts_eos_classifier.fc1.weight"]);
        _fc1B = WhisperOps.EnsureF32(w["tts_eos_classifier.fc1.bias"]);
        _fc2W = WhisperOps.EnsureF32(w["tts_eos_classifier.fc2.weight"]);
        _fc2B = WhisperOps.EnsureF32(w["tts_eos_classifier.fc2.bias"]);
    }

    /// <summary>Sigmoid stop probability for the last hidden state <paramref name="lastHidden"/> <c>[1,1,hidden]</c>.
    /// Upstream stops when this exceeds 0.5.</summary>
    public float StopProbability(IBackend backend, Tensor lastHidden)
    {
        if (_fc1W is null) throw new InvalidOperationException("VibeVoiceEosClassifier weights not loaded.");

        Tensor h1 = WhisperOps.ProjectLinear(backend, lastHidden, _fc1W!, _fc1B, 1, 1, _hidden, _hidden);
        backend.LeakyRelu(h1, h1, 0f);
        Tensor logit = WhisperOps.ProjectLinear(backend, h1, _fc2W!, _fc2B, 1, 1, _hidden, 1);
        h1.Dispose();
        float x = *(float*)logit.DataPointer;
        logit.Dispose();
        return 1f / (1f + MathF.Exp(-x));
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_fc1W, _fc1B, _fc2W, _fc2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
