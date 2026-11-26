using HartsyInference.Cuda;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>A <see cref="FactAttribute"/> reported as skipped (never passed) when no CUDA device opens.</summary>
public sealed class CudaFactAttribute : FactAttribute
{
    public CudaFactAttribute()
    {
        if (!CudaContext.IsAvailable()) Skip = "SKIPPED: no usable CUDA device; GPU check did not run.";
    }
}

/// <summary>A <see cref="TheoryAttribute"/> reported as skipped (never passed) when no CUDA device opens.</summary>
public sealed class CudaTheoryAttribute : TheoryAttribute
{
    public CudaTheoryAttribute()
    {
        if (!CudaContext.IsAvailable()) Skip = "SKIPPED: no usable CUDA device; GPU check did not run.";
    }
}

/// <summary>Real-weights fact that also needs a CUDA device; skipped (never passed) when either the weights or the device are missing.</summary>
public sealed class CudaRealWeightsFactAttribute : HartsyInference.Audio.Tests.VoiceIsolator.RealWeightsFactAttribute
{
    public CudaRealWeightsFactAttribute(params string[] environmentVariables) : base(environmentVariables)
    {
        if (Skip is null && !CudaContext.IsAvailable()) Skip = "SKIPPED: no usable CUDA device; GPU check did not run.";
    }
}

/// <summary>Real-weights fact that also needs a Vulkan device; skipped (never passed) when either the weights or the device are missing.</summary>
public sealed class VulkanRealWeightsFactAttribute : HartsyInference.Audio.Tests.VoiceIsolator.RealWeightsFactAttribute
{
    public VulkanRealWeightsFactAttribute(params string[] environmentVariables) : base(environmentVariables)
    {
        if (Skip is null && !VulkanContext.IsAvailable()) Skip = "SKIPPED: no usable Vulkan device; GPU check did not run.";
    }
}

/// <summary>A <see cref="FactAttribute"/> reported as skipped (never passed) when no Vulkan device opens.</summary>
public sealed class VulkanFactAttribute : FactAttribute
{
    public VulkanFactAttribute()
    {
        if (!VulkanContext.IsAvailable()) Skip = "SKIPPED: no usable Vulkan device; GPU check did not run.";
    }
}

/// <summary>A <see cref="TheoryAttribute"/> reported as skipped (never passed) when no Vulkan device opens.</summary>
public sealed class VulkanTheoryAttribute : TheoryAttribute
{
    public VulkanTheoryAttribute()
    {
        if (!VulkanContext.IsAvailable()) Skip = "SKIPPED: no usable Vulkan device; GPU check did not run.";
    }
}
