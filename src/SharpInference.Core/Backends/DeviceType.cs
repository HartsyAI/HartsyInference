namespace SharpInference.Core.Backends;

/// <summary>The type of compute device. CPU is always available. CUDA requires NVIDIA GPU + driver. Vulkan requires any GPU with Vulkan driver.</summary>
public enum DeviceType : byte
{
    /// <summary>CPU compute via SIMD kernels (AVX2/AVX-512/NEON).</summary>
    Cpu = 0,

    /// <summary>NVIDIA CUDA GPU compute via PTX kernels + cuBLAS (dotLLM pattern).</summary>
    Cuda = 1,

    /// <summary>Vulkan GPU compute via SPIR-V compute shaders (extends dotLLM's P/Invoke approach).</summary>
    Vulkan = 2,
}
