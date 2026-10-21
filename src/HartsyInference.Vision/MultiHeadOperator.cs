using System.Collections.Concurrent;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision;

/// <summary>The attention layer behind a vision site that permutes its own head-major operands: equal heads, F32, no weights, one per
/// head geometry, shared because the layer holds no state.</summary>
internal static class MultiHeadOperator
{
    private static readonly ConcurrentDictionary<(int Heads, int HeadDim), AttentionLayer> Layers = new();

    public static AttentionLayer For(int heads, int headDim) => Layers.GetOrAdd((heads, headDim), static key => new AttentionLayer(
        new AttentionLayerConfig { Heads = key.Heads, KvHeads = key.Heads, HeadDim = key.HeadDim, ComputeDType = DType.F32 }));
}
