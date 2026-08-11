using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>Per-utterance carried state for <see cref="HiFTNetVocoder.ForwardStreaming"/>: the growing mel
/// history and how many audio samples have already been emitted. Create one per utterance, feed it every mel
/// chunk of that utterance in order, dispose it when the utterance is done.
///
/// <para>Unlike Mimi's SEANet decoder (trained causal, so it gets a real incremental conv-tail/OLA-tail
/// design — see <c>MimiSeanetDecoderStreamState</c>), HiFTNetVocoder was trained with ordinary centered/
/// symmetric padding (non-causal). But every stage in its pipeline (F0 predictor, conv_pre/upsample/MRF,
/// harmonic-source generation, forward+inverse STFT) is purely local/convolutional/windowed — no attention,
/// no global normalization — so re-running the unmodified <see cref="HiFTNetVocoder.Forward"/> over the FULL
/// mel-so-far and only emitting the newly-settled SUFFIX of its output (holding back a safety margin near the
/// "now" edge, where the true future mel isn't known yet) reproduces the monolithic call's audio bit-for-bit
/// for every sample it emits. No separate causal reimplementation, no hand-derived per-layer receptive-field
/// arithmetic, no drift risk. The tradeoff is recompute cost (O(mel²) over an utterance instead of O(mel)) —
/// accepted for a correctness-first build, matching this vocoder's own existing host-side-DSP
/// TODO(gpu-residency) markers.</para></summary>
public sealed class HiFTStreamState : IDisposable
{
    internal Tensor? MelHistory;
    internal int EmittedSamples;
    private int _disposed;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        MelHistory?.Dispose();
    }
}
