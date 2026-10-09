namespace HartsyInference.Engine.Placement;

/// <summary>Where a model component's weights live while it runs.</summary>
public enum ResidencyMode
{
    /// <summary>Held on a device for the whole load.</summary>
    Resident,

    /// <summary>Read from storage each time it is used.</summary>
    Streamed,

    /// <summary>Held in host memory for the whole load.</summary>
    HostResident,

    /// <summary>Executed on the host from its stored form.</summary>
    CpuExecute,
}
