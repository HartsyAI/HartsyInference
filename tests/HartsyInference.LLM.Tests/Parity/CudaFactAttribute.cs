using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>A <see cref="FactAttribute"/> reported as skipped (never passed) when no CUDA device opens or the block-op PTX is missing.</summary>
public sealed class CudaFactAttribute : FactAttribute
{
    public CudaFactAttribute()
    {
        if (!DeviceProbe.CudaReady) Skip = $"SKIPPED: no usable CUDA device or block-op PTX ({DeviceProbe.CudaFailure}); GPU check did not run.";
    }
}
