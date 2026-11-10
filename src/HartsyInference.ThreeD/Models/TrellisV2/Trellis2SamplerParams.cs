using System.Text.Json;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>Parameters of one <c>FlowEulerGuidanceIntervalSampler</c> run: the pipeline <c>params</c> plus the sampler <c>sigma_min</c>.</summary>
public sealed record Trellis2SamplerParams
{
    /// <summary>Euler steps (<c>steps</c>).</summary>
    public required int Steps { get; init; }

    /// <summary>Classifier-free guidance strength (<c>guidance_strength</c>); 1 is the conditional prediction alone.</summary>
    public required float GuidanceStrength { get; init; }

    /// <summary>Weight of the std-matched rescale of the guided <c>x0</c> (<c>guidance_rescale</c>); 0 disables it.</summary>
    public required float GuidanceRescale { get; init; }

    /// <summary>Guidance applies while <c>low &lt;= t &lt;= high</c> (<c>guidance_interval</c>).</summary>
    public required double IntervalLow { get; init; }

    public required double IntervalHigh { get; init; }

    /// <summary>Time-warp factor of the schedule (<c>rescale_t</c>).</summary>
    public required double RescaleT { get; init; }

    /// <summary>Minimum noise scale of the flow (<c>sigma_min</c>).</summary>
    public double SigmaMin { get; init; } = 1e-5;

    /// <summary>Throws <see cref="ArgumentException"/> when the parameters cannot drive a sampler.</summary>
    public void Validate()
    {
        if (Steps <= 0)
            throw new ArgumentException($"Sampler steps must be positive; got {Steps}.");
        if (!float.IsFinite(GuidanceStrength) || !float.IsFinite(GuidanceRescale) || GuidanceRescale < 0f || GuidanceRescale > 1f)
            throw new ArgumentException("Guidance strength must be finite and the rescale weight within [0, 1].");
        if (!(RescaleT > 0.0) || !double.IsFinite(RescaleT))
            throw new ArgumentException($"rescale_t must be positive; got {RescaleT}.");
        if (!(SigmaMin > 0.0 && SigmaMin < 1.0))
            throw new ArgumentException($"sigma_min must be in (0, 1); got {SigmaMin}.");
        if (IntervalLow > IntervalHigh)
            throw new ArgumentException($"Guidance interval [{IntervalLow}, {IntervalHigh}] is empty.");
    }

    /// <summary>Reads a sampler node of <c>pipeline.json</c>: <c>args.sigma_min</c> and <c>params</c>.</summary>
    internal static Trellis2SamplerParams FromNode(JsonElement node)
    {
        string name = node.GetProperty("name").GetString() ?? string.Empty;
        if (name != "FlowEulerGuidanceIntervalSampler")
            throw new NotSupportedException($"Sampler '{name}' is not implemented; only FlowEulerGuidanceIntervalSampler is.");
        JsonElement p = node.GetProperty("params");
        JsonElement interval = p.GetProperty("guidance_interval");
        Trellis2SamplerParams result = new()
        {
            Steps = p.GetProperty("steps").GetInt32(),
            GuidanceStrength = (float)p.GetProperty("guidance_strength").GetDouble(),
            GuidanceRescale = (float)p.GetProperty("guidance_rescale").GetDouble(),
            IntervalLow = interval[0].GetDouble(),
            IntervalHigh = interval[1].GetDouble(),
            RescaleT = p.GetProperty("rescale_t").GetDouble(),
            SigmaMin = node.GetProperty("args").GetProperty("sigma_min").GetDouble(),
        };
        result.Validate();
        return result;
    }
}
