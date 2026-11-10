using HartsyInference.Cuda;
using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests.Parity;

internal static class DeviceProbe
{
    public static bool CudaReady => Cached.CudaFailure is null;
    public static bool VulkanReady => Cached.VulkanFailure is null;
    public static string? CudaFailure => Cached.CudaFailure;
    public static string? VulkanFailure => Cached.VulkanFailure;

    private static class Cached
    {
        public static readonly string? CudaFailure = ProbeCuda();
        public static readonly string? VulkanFailure = ProbeVulkan();
    }

    // A probe: any failure to open the device or read its kernels means this lane is not available here; the reason is reported.
    private static string? ProbeCuda()
    {
        try
        {
            string ptx = CudaPtxDirectory.Resolve();
            if (!File.Exists(Path.Combine(ptx, "hadamard.ptx")) || !File.Exists(Path.Combine(ptx, "fp8_block_linear.ptx")))
                return $"missing in {ptx}";
            using CudaBackend backend = new(0, ptx);
            return null;
        }
        catch (Exception ex)
        {
            return $"{ex.GetType().Name}: {ex.Message}";
        }
    }

    private static string? ProbeVulkan()
    {
        try
        {
            string spv = Path.Combine(AppContext.BaseDirectory, "Spirv");
            if (!File.Exists(Path.Combine(spv, "hadamard.spv")) || !File.Exists(Path.Combine(spv, "fp8_block_linear.spv")))
                return "hadamard.spv / fp8_block_linear.spv not built";
            using VulkanBackend backend = new();
            return null;
        }
        catch (Exception ex)
        {
            return $"{ex.GetType().Name}: {ex.Message}";
        }
    }
}
