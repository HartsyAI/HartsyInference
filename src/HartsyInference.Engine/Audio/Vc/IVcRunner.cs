using HartsyInference.Core.Backends;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>A loaded voice-conversion model reduced to: source audio (plus an optional target voice) → re-voiced PCM
/// at <see cref="SampleRate"/>. Both inputs arrive at the descriptor's input sample rate.</summary>
internal interface IVcRunner : IDisposable
{
    /// <summary>Output sample rate in Hz.</summary>
    int SampleRate { get; }

    /// <summary>Re-voices <paramref name="sourceMono"/>. <paramref name="targetMono"/> is the target voice for models
    /// that condition on one (OpenVoice tone-color transfer); null for source-only models such as RVC, which carry
    /// the target voice in their trained weights.</summary>
    float[] Convert(IBackend backend, float[] sourceMono, float[]? targetMono, VoiceConversionRequest request);
}
