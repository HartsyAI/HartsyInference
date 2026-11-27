using HartsyInference.Core.Backends;
using HartsyInference.LLM.Multimodal.Streaming;
using HartsyInference.LLM.Sampling;

namespace HartsyInference.LLM.Models.MossVl;

/// <summary>A <see cref="MossVlDecodeSession"/> as the decoder of a live <see cref="RealtimeVlmSession"/>: each segment's frames are
/// independent images to the model, appended to its cross-attention cache.</summary>
public sealed class MossVlStreamingDecoder : IStreamingVlmDecoder
{
    private readonly MossVlDecodeSession _session;

    /// <summary>Opens an empty decode state for a live stream.</summary>
    /// <param name="limits">The session's text and vision caps; null uses the model's context and the default vision cap.</param>
    public MossVlStreamingDecoder(MossVlTextDecoder decoder, MossVlConfig config, IBackend backend, MossVlSessionLimits? limits = null)
    {
        _session = new MossVlDecodeSession(decoder, config, backend,
            limits ?? new MossVlSessionLimits(MaxTextTokens: config.Text.MaxPositionEmbeddings));
    }

    public int TextTokens => _session.TextTokens;

    public int VisionTokens => _session.VisionTokens;

    public float[] ForwardLogits(ReadOnlySpan<int> ids, IReadOnlyList<VisionFrameFeatures> frames, CancellationToken cancel) =>
        _session.ForwardLogits(ids, AsImages(frames), cancel);

    public int ForwardSample(ReadOnlySpan<int> ids, IReadOnlyList<VisionFrameFeatures> frames, SamplerChain sampler, List<int> history,
        CancellationToken cancel) =>
        _session.ForwardSample(ids, AsImages(frames), sampler, history, cancel);

    public void Dispose() => _session.Dispose();

    private static MossVlImageFeatures[] AsImages(IReadOnlyList<VisionFrameFeatures> frames)
    {
        MossVlImageFeatures[] images = new MossVlImageFeatures[frames.Count];
        for (int i = 0; i < images.Length; i++)
            images[i] = new MossVlImageFeatures(frames[i].MergedTokens, frames[i].MergedHeight, frames[i].MergedWidth);
        return images;
    }
}
