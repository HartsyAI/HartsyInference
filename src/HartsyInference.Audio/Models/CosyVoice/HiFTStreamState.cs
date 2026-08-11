using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>Per-utterance carried state for <see cref="HiFTNetVocoder.ForwardStreaming"/>. Create one per
/// utterance, feed it every mel chunk of that utterance in order, dispose it when the utterance is done.
///
/// <para>HiFTNetVocoder has two qualitatively different pieces, and they need two different streaming
/// strategies:</para>
///
/// <para><b>conv_pre/upsample/MRF/conv_post + iSTFT</b> — purely local/convolutional/windowed, no
/// cross-frame accumulation. Re-running the unmodified <see cref="HiFTNetVocoder"/>'s core over the FULL
/// mel-so-far (<see cref="MelHistory"/>) and only emitting the newly-settled suffix (holding back
/// <c>StreamMarginSamples</c> near the "now" edge, where future mel isn't known yet) reproduces the
/// monolithic call's audio bit-for-bit for every sample it emits — safe because this piece has no memory
/// beyond its own local receptive field. <see cref="EmittedSamples"/> tracks the audio-domain emission
/// cursor for this mechanism.</para>
///
/// <para><b>NSF harmonic source</b> — <c>NsfVocoderDsp.GenerateHarmonicSource</c>'s phase accumulator
/// (<c>cum[h]</c>, one running sum per harmonic) is an UNBOUNDED integral over F0 in time order. Recompute-
/// with-margin does NOT work here: tiny per-call floating-point differences at otherwise-"settled" F0
/// positions get integrated into that running sum and, inside <c>sin()</c>, do not decay with distance from
/// the current edge — they compound with utterance length instead. Verified empirically 2026-08-10/11:
/// recompute-with-margin gave relL2≈0.6-0.8 on utterances beyond a few seconds, on BOTH real CosyVoice2 mel
/// and pure synthetic random mel, converging toward correct only once the margin approached the entire
/// utterance length (i.e. no streaming at all).</para>
///
/// <para>The first fix attempt — consume each historical F0 value exactly once, into a carried phase/RNG,
/// never re-derive it — was necessary but NOT sufficient on its own, and the reason why is itself a real
/// finding: <c>F0Predictor.Forward</c>'s <c>backend.Conv1d</c> (cuDNN) output for a given INTERIOR position
/// (position 10 of a 60-frame window, position 200 of a 300-frame window — nowhere near either edge) still
/// varied by up to ~0.16 Hz depending on the window's TOTAL length, converging to the reference only once the
/// window approached the reference's own size. cuDNN's algorithm selection depends on overall tensor shape,
/// not just local content — so "settled by margin" never actually held for F0Predictor's GPU path, and
/// "consume exactly once" just meant baking in whichever slightly-wrong value that call's window happened to
/// produce, permanently. The real fix: <see cref="F0Predictor.ForwardHostCpu"/> — a naive host C#
/// reimplementation of F0Predictor with no such shape-dependent code path, used ONLY here (the GPU
/// <see cref="F0Predictor.Forward"/> is untouched and still used by <see cref="HiFTNetVocoder.Forward"/>'s
/// non-streaming path, where a single fresh computation has no cross-call consistency requirement to violate).
/// With that in place: F0 for a given historical mel frame is derived exactly once — the call where it first
/// crosses <see cref="HarmonicSettledMelFrames"/> — and immediately consumed into
/// <see cref="HarmonicPhase"/>/<see cref="NoiseRngState"/>, which are carried forward and never reset or
/// revisited. <see cref="HarmonicAudioSoFar"/> is the resulting audio-domain NSF signal, append-only.</para>
///
/// <para><b>Verified 2026-08-11, both fixes in place</b>: chunked <c>ForwardStreaming</c> vs. a matched
/// host-CPU-F0 monolithic reference (<see cref="HiFTNetVocoder.ForwardHostCpuF0"/>) is exact to relL2≈2e-3,
/// bounded and NOT growing with utterance length, at chunk sizes 1/2/6/12 over a real 600-frame (~12s) mel —
/// see <c>HiftStreamParityTests</c>. That residual relL2≈2e-3 is the x-path's own cuDNN shape-dependent
/// noise (same class as F0Predictor's, just far smaller since it doesn't feed an accumulator here), not a
/// correctness bug. A real CosyVoice2 generation (streamed vs. the shipped GPU-F0 <c>Forward</c>) transcribes
/// Whisper-identical and matches RMS/dynamic range closely, confirming the (larger, expected) host-CPU-vs-GPU
/// F0Predictor numeric gap is perceptually inaudible.</para></summary>
public sealed class HiFTStreamState : IDisposable
{
    internal Tensor? MelHistory;
    internal int EmittedSamples;

    /// <summary>One running phase sum per harmonic (<see cref="HiFTNetVocoder.Harmonics"/> entries) — mutated
    /// in place by <see cref="HartsyInference.Audio.Dsp.NsfVocoderDsp.GenerateHarmonicSourceChunk"/>, never reset.</summary>
    internal readonly double[] HarmonicPhase = new double[HiFTNetVocoder.Harmonics];

    /// <summary>Deterministic NSF-noise RNG state, threaded through the same way as <see cref="HarmonicPhase"/>
    /// (unused when <c>HIFT_DETERMINISTIC=1</c> disables NSF noise, but always kept valid).</summary>
    internal uint NoiseRngState = 0x9E3779B9u;

    /// <summary>The finalized (never-to-be-recomputed) NSF harmonic-source audio, append-only.</summary>
    internal readonly List<float> HarmonicAudioSoFar = new();

    /// <summary>How many leading mel frames' worth of F0 have already been consumed into
    /// <see cref="HarmonicPhase"/>/<see cref="HarmonicAudioSoFar"/>. Frames at or beyond this index get a
    /// throwaway (non-persisted) phase continuation each call, purely so the conv stack has something to
    /// inject there — never real output, never emitted.</summary>
    internal int HarmonicSettledMelFrames;

    private int _disposed;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        MelHistory?.Dispose();
    }
}
