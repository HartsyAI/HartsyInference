using HartsyInference.Audio.Tests.VoiceIsolator;
using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weights fact that also needs a CUDA device; reported as skipped (never passed) when the weights or the device are missing.
/// The device check is the shared probe in tests/Shared/GpuDeviceFacts.cs (the same one <c>CudaDeviceFactAttribute</c> uses).</summary>
public sealed class CudaRealWeightsFactAttribute : RealWeightsFactAttribute
{
    public CudaRealWeightsFactAttribute(params string[] environmentVariables) : base(environmentVariables)
    {
        if (Skip is null && GpuDevices.CudaFailure is { } failure) Skip = $"SKIPPED: no usable CUDA device ({failure}); GPU check did not run.";
    }
}

/// <summary>Real-weights fact that also needs a Vulkan device; reported as skipped (never passed) when the weights or the device are missing.
/// The device check is the shared probe in tests/Shared/GpuDeviceFacts.cs (the same one <c>VulkanDeviceFactAttribute</c> uses).</summary>
public sealed class VulkanRealWeightsFactAttribute : RealWeightsFactAttribute
{
    public VulkanRealWeightsFactAttribute(params string[] environmentVariables) : base(environmentVariables)
    {
        if (Skip is null && GpuDevices.VulkanFailure is { } failure) Skip = $"SKIPPED: no usable Vulkan device ({failure}); GPU check did not run.";
    }
}
