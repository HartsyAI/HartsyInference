namespace HartsyInference.Engine.Planning.Memory;

/// <summary>One phase of a generation: the component it keeps on the device and what it needs beside it.</summary>
/// <param name="Component">Which weights this phase holds.</param>
/// <param name="WeightBytes">Resident size of those weights on the target device.</param>
/// <param name="ActivationBytes">Working memory the phase needs beside its weights at this geometry.</param>
/// <param name="StreamFloorWeightBytes">Weights that must stay on the device when the phase streams (shared weights
/// plus the prefetch window). Only meaningful when <paramref name="Streamable"/> is true.</param>
/// <param name="Streamable">Whether the model wires block streaming for this phase.</param>
public readonly record struct MemoryPhase(MemoryComponent Component, long WeightBytes, long ActivationBytes,
    long StreamFloorWeightBytes, bool Streamable)
{
    /// <summary>Bytes the phase needs with its weights fully resident.</summary>
    public long ResidentBytes => WeightBytes + ActivationBytes;

    /// <summary>The least the phase can run in: streamed when it can stream, otherwise fully resident.</summary>
    public long FloorBytes => Streamable ? Math.Min(StreamFloorWeightBytes, WeightBytes) + ActivationBytes : ResidentBytes;
}
