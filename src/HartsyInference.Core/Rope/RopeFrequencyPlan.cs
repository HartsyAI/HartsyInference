namespace HartsyInference.Core.Rope;

/// <summary>Frozen per-pair frequencies, computed once and shared by every table build. Each pair turns at its own frequency and
/// reads one coordinate axis. <see cref="Create"/> gives every axis a contiguous run of pairs on its own schedule (the Flux
/// form); <see cref="CreateSectioned"/> runs one schedule over the whole head and assigns each pair an axis (the M-RoPE form).</summary>
public sealed class RopeFrequencyPlan
{
    private readonly int[] _axisDims;
    private readonly int[] _pairAxes;
    private readonly double[] _inverseFrequencies;
    private readonly double[] _denominators;

    private RopeFrequencyPlan(int[] axisDims, int[] pairAxes, double[] inverseFrequencies, double[] denominators)
    {
        _axisDims = axisDims;
        _pairAxes = pairAxes;
        HeadDim = 2 * pairAxes.Length;
        _inverseFrequencies = inverseFrequencies;
        _denominators = denominators;
    }

    /// <summary>Sum of the axis dims.</summary>
    public int HeadDim { get; }

    /// <summary>Number of coordinate axes.</summary>
    public int Axes => _axisDims.Length;

    /// <summary>Rotation pairs per token: <c>HeadDim / 2</c>.</summary>
    public int Pairs => HeadDim / 2;

    /// <summary>Channels that read each axis: twice the pairs assigned to it.</summary>
    public ReadOnlySpan<int> AxisDims => _axisDims;

    /// <summary>The axis each pair reads its coordinate from.</summary>
    internal ReadOnlySpan<int> PairAxes => _pairAxes;

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
            if (dim < 0 || (dim & 1) != 0)
                throw new ArgumentOutOfRangeException(nameof(spec), dim, "RoPE axis dims must be non-negative and even.");
            if (headDim > int.MaxValue - dim)
                throw new ArgumentOutOfRangeException(nameof(spec), dim, "RoPE head dimension overflows.");
            axisDims[a] = dim;
            pairOffsets[a] = headDim / 2;
            headDim += dim;
        }

        if (headDim == 0)
            throw new ArgumentException("RoPE needs a positive head dimension.", nameof(spec));
        int pairs = headDim / 2;
        int[] pairAxes = new int[pairs];
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
                pairAxes[offset + k] = a;
                denominators[offset + k] = denominator;
                inverse[offset + k] = 1.0 / denominator;
            }
        }

        return new RopeFrequencyPlan(axisDims, pairAxes, inverse, denominators);
    }

    /// <summary>One frequency schedule over the whole head, M-RoPE style: pair <c>k</c> turns at
    /// <c>1 / theta^(2k / headDim)</c> and reads the coordinate of axis <c>pairAxes[k]</c>, so the head dimension is twice
    /// the length of <paramref name="pairAxes"/>. Which pairs read which axis is the model's rule (contiguous sections,
    /// an interleaved pattern); an axis no pair reads gets a zero <see cref="AxisDims"/> entry.</summary>
    public static RopeFrequencyPlan CreateSectioned(double theta, ReadOnlySpan<int> pairAxes, int axes)
    {
        if (!double.IsFinite(theta) || theta <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(theta), theta, "RoPE base must be finite and positive.");
        if (axes < 1)
            throw new ArgumentOutOfRangeException(nameof(axes), axes, "RoPE needs at least one axis.");
        if (pairAxes.IsEmpty)
            throw new ArgumentException("RoPE needs a positive head dimension.", nameof(pairAxes));
        if (pairAxes.Length > int.MaxValue / 2)
            throw new ArgumentOutOfRangeException(nameof(pairAxes), pairAxes.Length, "RoPE head dimension overflows.");

        int pairs = pairAxes.Length;
        int headDim = 2 * pairs;
        int[] axisDims = new int[axes];
        int[] frozenAxes = new int[pairs];
        double[] inverse = new double[pairs];
        double[] denominators = new double[pairs];
        for (int k = 0; k < pairs; k++)
        {
            int axis = pairAxes[k];
            if ((uint)axis >= (uint)axes)
                throw new ArgumentOutOfRangeException(nameof(pairAxes), axis, $"Pair {k} names axis {axis}; the plan has {axes}.");
            frozenAxes[k] = axis;
            axisDims[axis] += 2;
            double denominator = Math.Pow(theta, (double)(2 * k) / headDim);
            denominators[k] = denominator;
            inverse[k] = 1.0 / denominator;
        }
        return new RopeFrequencyPlan(axisDims, frozenAxes, inverse, denominators);
    }
}
