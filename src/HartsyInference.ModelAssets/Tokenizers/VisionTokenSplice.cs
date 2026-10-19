using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Writes vision-tower tokens over a prompt's placeholder embeddings (transformers' <c>masked_scatter</c> on the
/// image token): the <c>k</c>-th placeholder row takes the <c>k</c>-th row of the images' features, in order.</summary>
public static class VisionTokenSplice
{
    /// <summary>Overwrites the <paramref name="imageTokenId"/> rows of host F32 <paramref name="embeds"/>
    /// <c>[1, ids.Length, hidden]</c> with the rows of host F32 <paramref name="imageFeatures"/> (each <c>[n_i, hidden]</c>
    /// or <c>[1, n_i, hidden]</c>), and returns the rows written. The features must fill the placeholders exactly.</summary>
    public static unsafe int Apply(Tensor embeds, ReadOnlySpan<int> ids, int imageTokenId, IReadOnlyList<Tensor> imageFeatures)
    {
        ArgumentNullException.ThrowIfNull(embeds);
        ArgumentNullException.ThrowIfNull(imageFeatures);
        int hidden = (int)embeds.Shape[embeds.Shape.Rank - 1];
        if (embeds.DType != DType.F32 || embeds.ElementCount != (long)ids.Length * hidden)
            throw new ArgumentException($"Embeddings {embeds.Shape} {embeds.DType} do not hold {ids.Length} F32 rows.", nameof(embeds));
        long featureRows = 0;
        foreach (Tensor features in imageFeatures)
        {
            ArgumentNullException.ThrowIfNull(features);
            if (features.DType != DType.F32 || features.Shape[features.Shape.Rank - 1] != hidden)
                throw new ArgumentException($"Image features {features.Shape} {features.DType} are not F32 rows of width {hidden}.", nameof(imageFeatures));
            featureRows += features.ElementCount / hidden;
        }
        int placeholders = 0;
        foreach (int id in ids)
        {
            if (id == imageTokenId) placeholders++;
        }
        if (placeholders != featureRows)
            throw new ArgumentException($"The prompt holds {placeholders} image placeholders for {featureRows} vision tokens.", nameof(ids));
        float* dst = (float*)embeds.DataPointer;
        long rowBytes = (long)hidden * sizeof(float);
        int image = 0;
        long row = 0, rowsInImage = imageFeatures.Count > 0 ? imageFeatures[0].ElementCount / hidden : 0;
        for (int s = 0; s < ids.Length; s++)
        {
            if (ids[s] != imageTokenId)
                continue;
            while (row == rowsInImage)
            {
                image++;
                row = 0;
                rowsInImage = imageFeatures[image].ElementCount / hidden;
            }
            float* src = (float*)imageFeatures[image].DataPointer + row * hidden;
            Buffer.MemoryCopy(src, dst + (long)s * hidden, rowBytes, rowBytes);
            row++;
        }
        return placeholders;
    }
}
