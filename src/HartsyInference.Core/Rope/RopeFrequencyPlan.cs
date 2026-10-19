namespace HartsyInference.Core.Rope;

/// <summary>Frozen per-pair frequencies for a <see cref="RopeSpec"/>, computed once and shared by every table build.</summary>
public sealed class RopeFrequencyPlan
{
    private readonly int[] _axisDims;
    private readonly int[] _axisPairOffsets;
    private readonly double[] _inverseFrequencies;
    private readonly double[] _denominators;

    private RopeFrequencyPlan(int[] axisDims, int[] axisPairOffsets, int headDim, double[] inverseFrequencies, double[] denominators)
    {
        _axisDims = axisDims;
        _axisPairOffsets = axisPairOffsets;
        HeadDim = headDim;
        _inverseFrequencies = inverseFrequencies;
        _denominators = denominators;
    }

    /// <summary>Sum of the axis dims.</summary>
    public int HeadDim { get; }

    /// <summary>Number of coordinate axes.</summary>
    public int Axes => _axisDims.Length;

    /// <summary>Rotation pairs per token: <c>HeadDim / 2</c>.</summary>
    public int Pairs => HeadDim / 2;

    /// <summary>Channel count per axis.</summary>
    public ReadOnlySpan<int> AxisDims => _axisDims;

    internal ReadOnlySpan<int> AxisPairOffsets => _axisPairOffsets;

    internal ReadOnlySpan<double> InverseFrequencies => _inverseFrequencies;

    internal ReadOnlySpan<double> Denominators => _denominators;

    /// <summary>Validates <paramref name="spec"/> and computes its frequencies. Later builds use the frozen result.</summary>
    public static RopeFrequencyPlan Create(RopeSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        if (spec.AxisDims is null || spec.AxisDims.Count == 0)
            throw new ArgumentException("RoPE needs at least one axis.", nameof(spec));
        if (!double.IsFinite(spec.Theta) || spec.Theta <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(spec), spec.Theta, "RoPE base must be finite and positive.");
        int axes = spec.AxisDims.Count;
        if (spec.NtkScales is not null && spec.NtkScales.Count != axes)
            throw new ArgumentException("NtkScales needs one entry per axis.", nameof(spec));

        int[] axisDims = new int[axes];
        int[] pairOffsets = new int[axes];
        int headDim = 0;
        for (int a = 0; a < axes; a++)
        {
            int dim = spec.AxisDims[a];
            if (dim <= 0 || (dim & 1) != 0)
                throw new ArgumentOutOfRangeException(nameof(spec), dim, "RoPE axis dims must be positive and even.");
            if (headDim > int.MaxValue - dim)
                throw new ArgumentOutOfRangeException(nameof(spec), dim, "RoPE head dimension overflows.");
            axisDims[a] = dim;
            pairOffsets[a] = headDim / 2;
            headDim += dim;
        }

        int pairs = headDim / 2;
        double[] inverse = new double[pairs];
        double[] denominators = new double[pairs];
        for (int a = 0; a < axes; a++)
        {
            int dim = axisDims[a];
            double scale = spec.NtkScales is null ? 1.0 : spec.NtkScales[a];
            if (!double.IsFinite(scale) || scale <= 0.0)
                throw new ArgumentOutOfRangeException(nameof(spec), scale, "NTK scales must be finite and positive.");
            double theta = spec.Theta * RopeFrequencyBuilder.NtkThetaFactor(scale, dim);
            if (!double.IsFinite(theta) || theta <= 0.0)
                throw new ArgumentOutOfRangeException(nameof(spec), theta, "Scaled RoPE base must be finite and positive.");

            int offset = pairOffsets[a];
            for (int k = 0; k < dim / 2; k++)
            {
                double denominator = Math.Pow(theta, (double)(2 * k) / dim);
                denominators[offset + k] = denominator;
                inverse[offset + k] = 1.0 / denominator;
            }
        }

        return new RopeFrequencyPlan(axisDims, pairOffsets, headDim, inverse, denominators);
    }
}
