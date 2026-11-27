namespace HartsyInference.Core.Backends;

/// <summary>One device-memory reading: the driver's free and total bytes, the stream-ordered pool's reserved and handed-out
/// bytes, and the bytes the residency cache holds for weights (with their cached casts) and for activations. A figure the
/// backend cannot report is -1.</summary>
public readonly record struct DeviceMemoryUsage(long FreeBytes, long TotalBytes, long PoolReservedBytes, long PoolUsedBytes,
    long WeightBytes, long ActivationBytes);
