namespace SharpInference.Core.Backends;

/// <summary>Identifies the device a tensor resides on or a backend targets. Following dotLLM's pattern of device-per-tensor tracking with ordinal support.</summary>
public readonly record struct DeviceKind(DeviceType Type, int Ordinal = 0)
{
    /// <summary>CPU device.</summary>
    public static readonly DeviceKind Cpu = new(DeviceType.Cpu, 0);

    /// <summary>First CUDA GPU (device 0).</summary>
    public static readonly DeviceKind Cuda0 = new(DeviceType.Cuda, 0);

    /// <summary>First Vulkan GPU (device 0).</summary>
    public static readonly DeviceKind Vulkan0 = new(DeviceType.Vulkan, 0);

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

    /// <summary>Whether this is any GPU device (CUDA or Vulkan).</summary>
    public bool IsGpu => Type == DeviceType.Cuda || Type == DeviceType.Vulkan;

    public override string ToString() => Type switch
    {
        DeviceType.Cpu => "cpu",
        DeviceType.Cuda => $"cuda:{Ordinal}",
        DeviceType.Vulkan => $"vulkan:{Ordinal}",
        _ => $"unknown:{Ordinal}",
    };
}
