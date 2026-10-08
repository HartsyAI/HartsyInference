using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe;

/// <summary>The default: resident experts run on the device, misses run on the host, nothing is uploaded to serve a miss.</summary>
public sealed class ResidentFirstPolicy : IMissExecutionPolicy
{
    /// <summary>Shared instance; the policy is stateless.</summary>
    public static ResidentFirstPolicy Instance { get; } = new();

    /// <inheritdoc/>
    public ExpertPlacement Place(ExpertKey key, bool resident, int rows) => resident ? ExpertPlacement.Gpu : ExpertPlacement.Cpu;
}
