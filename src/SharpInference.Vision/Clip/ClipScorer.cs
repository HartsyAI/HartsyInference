using SharpInference.Vision.Embeddings;

namespace SharpInference.Vision.Clip;

/// <summary>Cosine-similarity scoring over already-L2-normalized embeddings. Since
/// <see cref="ClipImageEncoder"/> and <see cref="ClipTextEmbedder"/> normalize their outputs,
/// the scoring head is just a dot product — no extra norm needed at score time.
/// <para><b>What about <c>logit_scale</c>?</b> OpenAI CLIP exposes a learned temperature
/// (<c>exp(logit_scale)</c>, typically ~100) and the contrastive loss multiplies cosine sim by it
/// before softmax. That's a training/classification convenience — for raw similarity scoring it
/// only rescales the range. We return the unscaled cosine similarity in <c>[-1, 1]</c>; callers
/// that want the softmax-classification interface can compose it themselves.</para></summary>
public static class ClipScorer
{
    /// <summary>Cosine similarity between two embeddings. Both must come from compatible
    /// encoders (same <see cref="ImageEmbedding.EmbeddingDim"/>) — comparing CLIP-L (768-dim)
    /// against CLIP-bigG (1280-dim) is meaningless.</summary>
    public static float Score(ImageEmbedding image, TextEmbedding text)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(text);
        if (image.EmbeddingDim != text.EmbeddingDim)
            throw new ArgumentException($"Embedding dims differ: image {image.EmbeddingDim} vs text {text.EmbeddingDim}. Both must come from the same CLIP preset.");
        return DotProduct(image.AsSpan(), text.AsSpan());
    }

    /// <summary>Cosine similarity between two image embeddings (image-image retrieval).</summary>
    public static float Score(ImageEmbedding a, ImageEmbedding b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.EmbeddingDim != b.EmbeddingDim)
            throw new ArgumentException($"Embedding dims differ: {a.EmbeddingDim} vs {b.EmbeddingDim}.");
        return DotProduct(a.AsSpan(), b.AsSpan());
    }

    /// <summary>Cosine similarity between two text embeddings.</summary>
    public static float Score(TextEmbedding a, TextEmbedding b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.EmbeddingDim != b.EmbeddingDim)
            throw new ArgumentException($"Embedding dims differ: {a.EmbeddingDim} vs {b.EmbeddingDim}.");
        return DotProduct(a.AsSpan(), b.AsSpan());
    }

    /// <summary>Returns a similarity matrix: <c>scores[i, j] = cos(images[i], texts[j])</c>.
    /// Convenient for zero-shot classification where you score one image against many candidate
    /// labels, or rank many images against one query.</summary>
    public static float[,] ScoreMatrix(IReadOnlyList<ImageEmbedding> images, IReadOnlyList<TextEmbedding> texts)
    {
        ArgumentNullException.ThrowIfNull(images);
        ArgumentNullException.ThrowIfNull(texts);
        if (images.Count == 0 || texts.Count == 0)
            return new float[images.Count, texts.Count];

        int dim = images[0].EmbeddingDim;
        for (int i = 0; i < images.Count; i++)
            if (images[i].EmbeddingDim != dim)
                throw new ArgumentException($"All images must share embedding dim {dim}; index {i} has {images[i].EmbeddingDim}.", nameof(images));
        for (int j = 0; j < texts.Count; j++)
            if (texts[j].EmbeddingDim != dim)
                throw new ArgumentException($"All texts must share embedding dim {dim}; index {j} has {texts[j].EmbeddingDim}.", nameof(texts));

        float[,] scores = new float[images.Count, texts.Count];
        for (int i = 0; i < images.Count; i++)
        {
            ReadOnlySpan<float> imgVec = images[i].AsSpan();
            for (int j = 0; j < texts.Count; j++)
            {
                scores[i, j] = DotProduct(imgVec, texts[j].AsSpan());
            }
        }
        return scores;
    }

    /// <summary>Convenience: rank candidate texts by similarity to one image, returning the top-K
    /// indices and scores in descending order. For zero-shot classification, pass the candidate
    /// class names (each embedded as text) and read the top match.</summary>
    public static IReadOnlyList<(int Index, float Score)> TopK(ImageEmbedding image, IReadOnlyList<TextEmbedding> candidates, int k)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(candidates);
        if (k <= 0)
            throw new ArgumentOutOfRangeException(nameof(k), k, "k must be positive.");
        if (candidates.Count == 0)
            return Array.Empty<(int, float)>();

        ReadOnlySpan<float> imgVec = image.AsSpan();
        (int Index, float Score)[] all = new (int, float)[candidates.Count];
        for (int i = 0; i < candidates.Count; i++)
        {
            if (candidates[i].EmbeddingDim != image.EmbeddingDim)
                throw new ArgumentException($"Candidate {i} has embedding dim {candidates[i].EmbeddingDim} != image {image.EmbeddingDim}.");
            all[i] = (i, DotProduct(imgVec, candidates[i].AsSpan()));
        }

        Array.Sort(all, (a, b) => b.Score.CompareTo(a.Score));
        int take = Math.Min(k, all.Length);
        (int, float)[] result = new (int, float)[take];
        Array.Copy(all, result, take);
        return result;
    }

    private static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"Span lengths differ: {a.Length} vs {b.Length}.");

        // Accumulate in F64 to keep precision on 1280-dim CLIP-G embeddings — F32 cumulative
        // rounding error becomes visible past ~1024 terms on near-unit-magnitude inputs.
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return (float)sum;
    }
}
