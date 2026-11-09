using HartsyInference.LLM.Multimodal.Streaming;
using HartsyInference.LLM.Sampling;

namespace HartsyInference.Engine.Services;

/// <summary>A streaming decoder whose every forward holds a device gate, so a long-lived session shares its GPU with the engine's other
/// work between tokens instead of holding the device for its whole life.</summary>
internal sealed class GatedStreamingDecoder(IStreamingVlmDecoder inner, Func<CancellationToken, IDisposable> gate) : IStreamingVlmDecoder
{
    public int TextTokens => inner.TextTokens;

    public int VisionTokens => inner.VisionTokens;

    public float[] ForwardLogits(ReadOnlySpan<int> ids, IReadOnlyList<VisionFrameFeatures> frames, CancellationToken cancel)
    {
        using IDisposable held = gate(cancel);
        return inner.ForwardLogits(ids, frames, cancel);
    }

    public int ForwardSample(ReadOnlySpan<int> ids, IReadOnlyList<VisionFrameFeatures> frames, SamplerChain sampler, List<int> history,
        CancellationToken cancel)
    {
        using IDisposable held = gate(cancel);
        return inner.ForwardSample(ids, frames, sampler, history, cancel);
    }

    public void Dispose() => inner.Dispose();
}
