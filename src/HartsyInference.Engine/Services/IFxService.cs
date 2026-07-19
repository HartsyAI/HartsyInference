using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed audio-effects surface: stem separation (Demucs) and speech enhancement (Resemble-Enhance).</summary>
public interface IFxService
{
    /// <summary>Separates the mix in <paramref name="request"/> into stems.</summary>
    Task<StemsResult> SeparateAsync(ModelSpec spec, FxSeparateRequest request, CancellationToken cancel = default);

    /// <summary>Enhances the audio in <paramref name="request"/>.</summary>
    Task<AudioResult> EnhanceAsync(ModelSpec spec, FxEnhanceRequest request, CancellationToken cancel = default);
}
