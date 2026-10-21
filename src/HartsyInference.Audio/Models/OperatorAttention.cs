using HartsyInference.Core.Layers.Attention;

namespace HartsyInference.Audio.Models;

/// <summary>The weightless layer of a site that builds its own head-major operands: it runs only the operator stage.</summary>
internal static class OperatorAttention
{
    /// <summary>A layer of <paramref name="heads"/> equal query and key/value heads; <paramref name="scale"/> 0 means <c>1/√headDim</c>.</summary>
    public static AttentionLayer Create(int heads, int headDim, float scale = 0f) =>
        new(new AttentionLayerConfig { Heads = heads, KvHeads = heads, HeadDim = headDim, Scale = scale });
}
