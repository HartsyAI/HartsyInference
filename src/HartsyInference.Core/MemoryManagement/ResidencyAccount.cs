namespace HartsyInference.Core.MemoryManagement;

/// <summary>Who holds a residency reservation. Pinned staging is charged to host-pinned memory; every other account to the device budget.</summary>
public enum ResidencyAccount
{
    DenseWeights,
    RoutedExperts,
    Engram,
    Vision,
    DraftHeads,
    KvCache,
    DraftState,
    Activations,
    ConversionWorkspace,
    CommWorkspace,
    PinnedStaging,
    Headroom,
}
