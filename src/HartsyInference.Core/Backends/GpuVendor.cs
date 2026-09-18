namespace HartsyInference.Core.Backends;

/// <summary>Who made the device a backend is running on, as the backend reports it.
///
/// <para>Separate from <see cref="DeviceType"/>, which says which API is driving the device: Vulkan runs on all of
/// these, so the API cannot answer questions whose answer is per-vendor. The ones that matter in practice are
/// cooperative-matrix reliability (which ggml handles with a per-vendor allow/deny list rather than trusting the
/// capability bit), subgroup width, and which kernel tuning constants to pick.</para>
///
/// <para><see cref="Software"/> is its own value rather than <see cref="Other"/> because a CPU implementation of a
/// GPU API — llvmpipe/lavapipe — is useful for correctness and worthless for performance, and a benchmark that
/// silently ran on one is worse than no benchmark.</para></summary>
public enum GpuVendor : byte
{
    /// <summary>Not a GPU, or the backend does not report a vendor.</summary>
    None = 0,

    /// <summary>NVIDIA (PCI vendor id 0x10DE).</summary>
    Nvidia = 1,

    /// <summary>AMD (0x1002).</summary>
    Amd = 2,

    /// <summary>Intel (0x8086).</summary>
    Intel = 3,

    /// <summary>Apple (0x106B).</summary>
    Apple = 4,

    /// <summary>A software rasterizer presenting itself as a GPU (llvmpipe/lavapipe, SwiftShader).</summary>
    Software = 5,

    /// <summary>A real GPU from a vendor this enum does not name — ARM Mali, Qualcomm Adreno, Imagination.</summary>
    Other = 6,
}
