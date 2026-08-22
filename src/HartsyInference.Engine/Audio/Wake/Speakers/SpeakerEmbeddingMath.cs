namespace HartsyInference.Engine.Audio.Wake.Speakers;

/// <summary>The vector arithmetic behind speaker profiles, deliberately free of any model, backend or disk dependency so enrollment and the open-set decision stay testable on a machine with no CAM++ weights.</summary>
public static class SpeakerEmbeddingMath
{
    /// <summary>Below this L2 norm a vector carries no usable direction, so cosine against it is undefined.</summary>
    private const double DegenerateNorm = 1e-9d;

    /// <summary>Speaker model for a set of enrollment utterances: each embedding is L2-normalized (a no-op when the caller already normalized), those unit vectors are averaged, and the mean is L2-normalized again. Normalizing the inputs first is what makes the centroid a direction average rather than a magnitude-weighted one — a loud utterance must not outvote a quiet one.</summary>
    public static float[] Centroid(IReadOnlyList<float[]> embeddings)
    {
        ArgumentNullException.ThrowIfNull(embeddings);
        if (embeddings.Count == 0)
        {
            throw new ArgumentException("A speaker centroid needs at least one enrollment embedding.", nameof(embeddings));
        }
        int dimension = embeddings[0]?.Length ?? 0;
        if (dimension == 0)
        {
            throw new ArgumentException("Enrollment embeddings must be non-empty.", nameof(embeddings));
        }

        double[] accumulator = new double[dimension];
        for (int i = 0; i < embeddings.Count; i++)
        {
            float[]? embedding = embeddings[i];
            if (embedding is null || embedding.Length != dimension)
            {
                throw new ArgumentException(
                    $"Enrollment embedding {i} has dimension {embedding?.Length ?? 0}, expected {dimension}.", nameof(embeddings));
            }
            double norm = Math.Sqrt(SquaredNorm(embedding));
            double inverse = norm < DegenerateNorm ? 0d : 1d / norm;
            for (int d = 0; d < dimension; d++)
            {
                accumulator[d] += embedding[d] * inverse;
            }
        }

        float[] centroid = new float[dimension];
        for (int d = 0; d < dimension; d++)
        {
            centroid[d] = (float)(accumulator[d] / embeddings.Count);
        }
        NormalizeInPlace(centroid);
        return centroid;
    }

    /// <summary>A unit-length copy; the source is left untouched.</summary>
    public static float[] Normalized(ReadOnlySpan<float> vector)
    {
        float[] copy = vector.ToArray();
        NormalizeInPlace(copy);
        return copy;
    }

    /// <summary>Scales to unit L2 length; a vector whose norm is numerically zero is left as-is rather than being blown up to infinity.</summary>
    public static void NormalizeInPlace(Span<float> vector)
    {
        double norm = Math.Sqrt(SquaredNorm(vector));
        if (norm < DegenerateNorm)
        {
            return;
        }
        float inverse = (float)(1d / norm);
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] *= inverse;
        }
    }

    /// <summary>Cosine similarity in [-1, 1], dividing by both norms so the result is correct for inputs that were never normalized. Returns 0 when either side is degenerate.</summary>
    public static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cosine similarity needs matching dimensions, got {a.Length} and {b.Length}.", nameof(b));
        }
        double dot = 0d;
        double normA = 0d;
        double normB = 0d;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            normA += (double)a[i] * a[i];
            normB += (double)b[i] * b[i];
        }
        double denominator = Math.Sqrt(normA) * Math.Sqrt(normB);
        if (denominator < DegenerateNorm)
        {
            return 0f;
        }
        return (float)Math.Clamp(dot / denominator, -1d, 1d);
    }

    private static double SquaredNorm(ReadOnlySpan<float> vector)
    {
        double sum = 0d;
        for (int i = 0; i < vector.Length; i++)
        {
            sum += (double)vector[i] * vector[i];
        }
        return sum;
    }
}
