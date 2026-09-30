namespace HartsyInference.Core.Engram;

/// <summary>Where an <see cref="EngramTableStore"/> keeps the rows it has read.</summary>
public enum EngramBacking
{
    /// <summary>Rows sit in device memory. Not supported yet: it needs the model's device buffers, which arrive with the Engram module.</summary>
    Device,

    /// <summary>The whole owned range is held in host memory and never evicted; the budget must cover it.</summary>
    HostResident,

    /// <summary>Rows stay on disk and are read on demand into a budget-bounded LRU cache in host memory.</summary>
    Storage,
}
