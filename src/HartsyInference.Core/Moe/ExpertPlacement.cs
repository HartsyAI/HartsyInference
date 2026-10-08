namespace HartsyInference.Core.Moe;

/// <summary>Where one routed expert's rows run. The GPU runs experts already resident in the cache; the CPU runs the rest.</summary>
public enum ExpertPlacement
{
    /// <summary>Runs on the device from a resident cache copy.</summary>
    Gpu,

    /// <summary>Runs on the host from the authoritative weights.</summary>
    Cpu,
}
