using HartsyInference.Core.IO;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Where one layer's Engram table lives in its pread-only shard; the table is far larger than host RAM and is read by row, never mapped.</summary>
/// <param name="Layer">Backbone layer that owns the table.</param>
/// <param name="Embed">The e4m3 payload tensor, <c>[rows, 256]</c>.</param>
/// <param name="Scale">The e8m0 scale tensor, <c>[rows, 8]</c>, or null when the producer has none.</param>
/// <param name="EmbedSource">Positional-read handle of the payload's shard (owned by the checkpoint; do not dispose).</param>
/// <param name="ScaleSource">Positional-read handle of the scale's shard, or null.</param>
public sealed record DeepSeekV41EngramTable(
    int Layer,
    TensorLocation Embed,
    TensorLocation? Scale,
    IWeightByteSource EmbedSource,
    IWeightByteSource? ScaleSource)
{
    /// <summary>Number of table rows.</summary>
    public long Rows => Embed.Shape[0];
}
