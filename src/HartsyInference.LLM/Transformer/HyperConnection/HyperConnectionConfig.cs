namespace HartsyInference.LLM.Transformer;

/// <summary>The formula parameters of one hyper-connection. A model difference is a field here, never a branch in <see cref="HyperConnection"/>.</summary>
public sealed record HyperConnectionConfig
{
    /// <summary>Residual copies.</summary>
    public required int Hc { get; init; }

    /// <summary>Residual width.</summary>
    public required int Dim { get; init; }

    /// <summary>Sinkhorn rounds on comb. Both models use 20; each reads it from its own config.</summary>
    public required int SinkhornIters { get; init; }

    /// <summary>Added to the comb normalizers.</summary>
    public required float SinkhornEps { get; init; }

    /// <summary>Added to the sigmoid pre gates and to the head gates. ERNIE keeps it apart from <see cref="SinkhornEps"/> (<c>hc_pre_eps</c>); both are 1e-6.</summary>
    public required float PreEps { get; init; }

    /// <summary>Multiplier on the sigmoid post gates. ERNIE's reference passes 2.0; DeepSeek's kernels fix 2.</summary>
    public required float PostMultiplier { get; init; }

    /// <summary>Epsilon of the RMS statistic over the flattened stream (<c>rms_norm_eps</c>).</summary>
    public required float NormEps { get; init; }

    /// <summary>RMS form for the sublayer mixes, see <see cref="RmsEpsPlacement"/>.</summary>
    public required RmsEpsPlacement MixRms { get; init; }

    /// <summary>RMS form for the head pre-mix. ERNIE's head uses <see cref="RmsEpsPlacement.InsideSqrt"/> while its mixes use the other form; DeepSeek has no head projection.</summary>
    public required RmsEpsPlacement HeadRms { get; init; }

    /// <summary>DeepSeek-V4.1 hyper-connection: one <c>hc_eps</c> for both gates, eps inside the sqrt, post multiplier 2.</summary>
    public static HyperConnectionConfig DeepSeekV41(int hc, int dim, int iters, float hcEps, float normEps) => new()
    {
        Hc = hc,
        Dim = dim,
        SinkhornIters = iters,
        SinkhornEps = hcEps,
        PreEps = hcEps,
        PostMultiplier = 2f,
        NormEps = normEps,
        MixRms = RmsEpsPlacement.InsideSqrt,
        HeadRms = RmsEpsPlacement.InsideSqrt,
    };

    /// <summary>ERNIE-5.5 hyper-connection: eps outside the sqrt for the mixes (vLLM <c>mhc_kernels.py</c>), post multiplier 2.0 (<c>hc_post_alpha</c>).</summary>
    public static HyperConnectionConfig Ernie55(int hc, int dim, int iters, float hcEps, float normEps) => new()
    {
        Hc = hc,
        Dim = dim,
        SinkhornIters = iters,
        SinkhornEps = hcEps,
        PreEps = hcEps,
        PostMultiplier = 2f,
        NormEps = normEps,
        MixRms = RmsEpsPlacement.OutsideSqrt,
        HeadRms = RmsEpsPlacement.InsideSqrt,
    };
}
