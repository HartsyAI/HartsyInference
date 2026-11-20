using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>A parameterized CUDA check with the same resource guard as CudaFact.</summary>
public sealed class CudaTheoryAttribute : TheoryAttribute
{
    public CudaTheoryAttribute()
    {
        if (!DeviceProbe.CudaReady) Skip = $"SKIPPED: no usable CUDA device or PTX ({DeviceProbe.CudaFailure}).";
    }
}
