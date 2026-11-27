namespace HartsyInference.Core.Imaging;

/// <summary>Per-channel mean and standard deviation that turn OKLab into the values a model trains on and back. There is
/// no default: the numbers belong to the model's training data, and <c>default</c> is refused when it is used. Two
/// normalizations built from the same numbers are equal.</summary>
public readonly record struct OkLabNormalization
{
    /// <summary>Creates a normalization from the (L, a, b) means and standard deviations.</summary>
    public OkLabNormalization(ReadOnlySpan<float> mean, ReadOnlySpan<float> std)
    {
        if (mean.Length != 3 || std.Length != 3)
            throw new ArgumentException("Mean and standard deviation each need three values (L, a, b).");
        foreach (float value in std)
        {
            if (!(value > 0f) || !float.IsFinite(value))
                throw new ArgumentException("Every standard deviation must be positive and finite.", nameof(std));
        }
        MeanL = mean[0];
        MeanA = mean[1];
        MeanB = mean[2];
        StdL = std[0];
        StdA = std[1];
        StdB = std[2];
    }

    /// <summary>The L channel's mean.</summary>
    public float MeanL { get; }

    /// <summary>The a channel's mean.</summary>
    public float MeanA { get; }

    /// <summary>The b channel's mean.</summary>
    public float MeanB { get; }

    /// <summary>The L channel's standard deviation.</summary>
    public float StdL { get; }

    /// <summary>The a channel's standard deviation.</summary>
    public float StdA { get; }

    /// <summary>The b channel's standard deviation.</summary>
    public float StdB { get; }

    /// <summary>Planar OKLab to <c>(value - mean) / std</c> per channel, in place.</summary>
    public void Normalize(Span<float> okLab, int pixels)
    {
        CheckReady(okLab.Length, pixels);
        for (int channel = 0; channel < 3; channel++)
        {
            Span<float> plane = okLab.Slice(channel * pixels, pixels);
            (float mean, float std) = Channel(channel);
            for (int i = 0; i < plane.Length; i++) plane[i] = (plane[i] - mean) / std;
        }
    }

    /// <summary>Planar normalized OKLab back to OKLab, in place.</summary>
    public void Denormalize(Span<float> normalized, int pixels)
    {
        CheckReady(normalized.Length, pixels);
        for (int channel = 0; channel < 3; channel++)
        {
            Span<float> plane = normalized.Slice(channel * pixels, pixels);
            (float mean, float std) = Channel(channel);
            for (int i = 0; i < plane.Length; i++) plane[i] = plane[i] * std + mean;
        }
    }

    private (float Mean, float Std) Channel(int channel) => channel switch
    {
        0 => (MeanL, StdL),
        1 => (MeanA, StdA),
        _ => (MeanB, StdB),
    };

    private void CheckReady(int length, int pixels)
    {
        if (!(StdL > 0f)) throw new InvalidOperationException("The normalization has no values; build it with the constructor.");
        if (length != 3L * pixels)
            throw new ArgumentException($"A planar span must hold 3 x {pixels} values; got {length}.");
    }
}
