namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>The fixed sin/cos embedding Hunyuan3D-2mv adds to every DINOv2 token of a view.</summary>
internal static class Hunyuan3DViewEmbedding
{
    /// <summary>The embedding of view index <paramref name="view"/> for <paramref name="width"/> channels: <c>sin(view * w)</c> then <c>cos(view * w)</c> with <c>w = 10000^(-i / (width / 2))</c>.</summary>
    internal static float[] Create(int view, int width)
    {
        if (width <= 0 || width % 2 != 0)
            throw new ArgumentOutOfRangeException(nameof(width), width, "Embedding width must be positive and even.");
        int half = width / 2;
        float[] embedding = new float[width];
        for (int i = 0; i < half; i++)
        {
            double omega = 1.0 / Math.Pow(10000.0, i / (double)half);
            double angle = view * omega;
            embedding[i] = (float)Math.Sin(angle);
            embedding[half + i] = (float)Math.Cos(angle);
        }

        return embedding;
    }
}
