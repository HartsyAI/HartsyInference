using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;

namespace HartsyInference.Cli.Infra;

/// <summary>Resolves the <c>--backend</c> selector (including <c>auto</c>) into a concrete <see cref="IBackend"/>,
/// centralizing the PTX/SPIR-V directory conventions that were copy-pasted across the sample CLIs.</summary>
public static class BackendFactory
{
    /// <summary>Subdirectory beside the executable holding compiled PTX kernels for the CUDA backend.</summary>
    public const string PtxDirName = "Ptx";

    /// <summary>Subdirectory beside the executable holding compiled SPIR-V kernels for the Vulkan backend.</summary>
    public const string SpirvDirName = "Spirv";

    /// <summary>Valid selector tokens accepted on the command line.</summary>
    public static IReadOnlyList<string> ValidSelectors { get; } = new[] { "auto", "cpu", "cuda", "vulkan" };

    /// <summary>Constructs the backend named by <paramref name="selector"/>, mapping <c>auto</c> via <see cref="Resolve"/>.</summary>
    public static IBackend Create(string selector)
    {
        string chosen = Resolve(selector);
        return chosen switch
        {
            "cuda" => new CudaBackend(deviceOrdinal: 0, ptxDir: Path.Combine(AppContext.BaseDirectory, PtxDirName)),
            "vulkan" => new VulkanBackend(deviceOrdinal: 0, spvDir: Path.Combine(AppContext.BaseDirectory, SpirvDirName)),
            "cpu" => new CpuBackend(),
            _ => throw new ArgumentException($"Unknown backend '{selector}'. Valid: {string.Join(", ", ValidSelectors)}."),
        };
    }

    /// <summary>Maps <c>auto</c> to the best available backend (CUDA when a device is present, else CPU) and passes
    /// explicit selectors through unchanged (lower-cased).</summary>
    public static string Resolve(string selector)
    {
        string s = (selector ?? "auto").Trim().ToLowerInvariant();
        if (s != "auto")
            return s;
        return CudaContext.IsAvailable() && CudaContext.GetDeviceCount() > 0 ? "cuda" : "cpu";
    }

    /// <summary>Human-readable description of what <paramref name="selector"/> resolves to, for banners and <c>--verbose</c>.</summary>
    public static string Describe(string selector)
    {
        string resolved = Resolve(selector);
        bool auto = string.Equals((selector ?? "auto").Trim(), "auto", StringComparison.OrdinalIgnoreCase);
        return auto ? $"auto → {resolved}" : resolved;
    }
}
