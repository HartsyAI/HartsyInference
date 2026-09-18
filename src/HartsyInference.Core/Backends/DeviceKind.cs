namespace HartsyInference.Core.Backends;

/// <summary>Identifies the device a tensor resides on or a backend targets. Following dotLLM's pattern of device-per-tensor tracking with ordinal support.</summary>
public readonly record struct DeviceKind(DeviceType Type, int Ordinal = 0)
{
    /// <summary>CPU device.</summary>
    public static readonly DeviceKind Cpu = new(DeviceType.Cpu, 0);

    /// <summary>Creates a CUDA device with the specified ordinal.</summary>
    public static DeviceKind Cuda(int ordinal = 0) => new(DeviceType.Cuda, ordinal);

    /// <summary>Creates a Vulkan device with the specified ordinal.</summary>
    public static DeviceKind Vulkan(int ordinal = 0) => new(DeviceType.Vulkan, ordinal);

    /// <summary>Whether this is a CPU device.</summary>
    public bool IsCpu => Type == DeviceType.Cpu;

    /// <summary>Whether this is a CUDA GPU device.</summary>
    public bool IsCuda => Type == DeviceType.Cuda;

    /// <summary>Whether this is a Vulkan GPU device.</summary>
    public bool IsVulkan => Type == DeviceType.Vulkan;

    /// <summary>Creates a ROCm device with the specified ordinal.</summary>
    public static DeviceKind Rocm(int ordinal = 0) => new(DeviceType.Rocm, ordinal);

    /// <summary>Creates a Metal device with the specified ordinal.</summary>
    public static DeviceKind Metal(int ordinal = 0) => new(DeviceType.Metal, ordinal);

    /// <summary>Whether this is any GPU device.</summary>
    /// <remarks>Asks what it means rather than listing the backends that exist, so a device added to
    /// <see cref="DeviceType"/> does not silently read as a CPU to every caller that gates on this.</remarks>
    public bool IsGpu => Type != DeviceType.Cpu;

    public override string ToString() => Type switch
    {
        DeviceType.Cpu => "cpu",
        DeviceType.Cuda => $"cuda:{Ordinal}",
        DeviceType.Vulkan => $"vulkan:{Ordinal}",
        DeviceType.Rocm => $"rocm:{Ordinal}",
        DeviceType.Metal => $"metal:{Ordinal}",
        _ => $"unknown:{Ordinal}",
    };
}
