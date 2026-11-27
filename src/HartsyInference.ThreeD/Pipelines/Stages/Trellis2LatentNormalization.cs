using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>The per-channel affine maps between a sampled latent and its <c>pipeline.json</c> statistics.</summary>
internal static class Trellis2LatentNormalization
{
    /// <summary>In place <c>x · std + mean</c> per channel over rows of <c>mean.Length</c> channels.</summary>
    internal static void Denormalize(Tensor latent, ReadOnlySpan<float> mean, ReadOnlySpan<float> std)
    {
        Span<float> values = Rows(latent, mean, std);
        int channels = mean.Length;
        for (int i = 0; i < values.Length; i++)
            values[i] = values[i] * std[i % channels] + mean[i % channels];
    }

    /// <summary>In place <c>(x - mean) / std</c> per channel over rows of <c>mean.Length</c> channels.</summary>
    internal static void Normalize(Tensor latent, ReadOnlySpan<float> mean, ReadOnlySpan<float> std)
    {
        Span<float> values = Rows(latent, mean, std);
        int channels = mean.Length;
        for (int i = 0; i < values.Length; i++)
            values[i] = (values[i] - mean[i % channels]) / std[i % channels];
    }

    private static Span<float> Rows(Tensor latent, ReadOnlySpan<float> mean, ReadOnlySpan<float> std)
    {
        Span<float> values = latent.AsSpan<float>();
        int channels = mean.Length;
        if (channels == 0 || std.Length != channels || values.Length % channels != 0)
            throw new ArgumentException($"A latent of {values.Length} values does not hold whole rows of {channels} channels.", nameof(latent));
        return values;
    }
}
