using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed voice-conversion surface (RVC source-only, OpenVoice tone-color transfer).</summary>
public interface IVoiceConversionService
{
    /// <summary>Re-voices the source audio in <paramref name="request"/>.</summary>
    Task<AudioResult> ConvertAsync(ModelSpec spec, VoiceConversionRequest request, CancellationToken cancel = default);
}
