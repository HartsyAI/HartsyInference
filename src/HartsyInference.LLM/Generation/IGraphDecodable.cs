using HartsyInference.Core.Backends;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Generation;

/// <summary>Optional capability of a model whose single-token greedy step can be captured as one CUDA graph and replayed.</summary>
internal interface IGraphDecodable
{
    /// <summary>True when this model's decode step is graph-capturable on <paramref name="backend"/>.</summary>
    bool SupportsGraphDecode(IBackend backend);

    /// <summary>Warms, makes resident and captures the decode step for <paramref name="state"/>, whose committed length is <paramref name="pos"/>; <paramref name="firstToken"/> is the token already sampled.</summary>
    GraphDecodeSession CaptureDecodeGraph(ISequenceState state, int pos, int firstToken, float repetitionPenalty, DeviceSamplerConfig? sampler = null);

    /// <summary>True when the captured step can draw tokens on the device (<paramref name="sampler"/> given to <see cref="CaptureDecodeGraph"/>) on <paramref name="backend"/>.</summary>
    bool SupportsDeviceSampling(IBackend backend) => false;

    /// <summary>Advances <paramref name="state"/> by one token after a replay, which writes KV on the device without moving the host cursor.</summary>
    void CommitReplayedStep(ISequenceState state);
}
