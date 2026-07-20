using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;

namespace HartsyInference.Engine;

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

    /// <summary>Explicit kernel-directory override; wins over auto-detection. For hosts that deploy the compiled
    /// kernels somewhere other than beside the engine assemblies.</summary>
    public static string? KernelDirOverride { get; set; }

    /// <summary>Directory holding compiled kernels, resolved relative to THIS assembly rather than the entry
    /// application: when the engine is hosted inside another app (a SwarmUI extension loaded into its own
    /// <c>AssemblyLoadContext</c>) <see cref="AppContext.BaseDirectory"/> is the host's bin dir, not the engine's,
    /// and the kernels ship beside the engine DLLs. Falls back to <see cref="AppContext.BaseDirectory"/> when the
    /// assembly location is unavailable (single-file publish) or the resolved directory does not exist.</summary>
    public static string KernelDir(string subdir)
    {
        if (!string.IsNullOrWhiteSpace(KernelDirOverride))
        {
            return Path.Combine(KernelDirOverride, subdir);
        }
        string asmDir = Path.GetDirectoryName(typeof(BackendFactory).Assembly.Location) ?? "";
        string candidate = Path.Combine(string.IsNullOrEmpty(asmDir) ? AppContext.BaseDirectory : asmDir, subdir);
        if (!Directory.Exists(candidate))
        {
            string fallback = Path.Combine(AppContext.BaseDirectory, subdir);
            if (Directory.Exists(fallback))
            {
                return fallback;
            }
        }
        return candidate;
    }

    /// <summary>Constructs a CUDA backend on <paramref name="ordinal"/> with the resolved PTX directory. Single place,
    /// so every caller (facade, TextService, recipes) agrees on where kernels live.</summary>
    public static IBackend CreateCuda(int ordinal) => new CudaBackend(ordinal, KernelDir(PtxDirName));

    /// <summary>Constructs a Vulkan backend on <paramref name="ordinal"/> with the resolved SPIR-V directory.</summary>
    public static IBackend CreateVulkan(int ordinal) => new VulkanBackend(ordinal, KernelDir(SpirvDirName));

    /// <summary>Constructs the backend named by <paramref name="selector"/>, mapping <c>auto</c> via <see cref="Resolve"/>.</summary>
    public static IBackend Create(string selector)
    {
        string chosen = Resolve(selector);
        return chosen switch
        {
            "cuda" => CreateCuda(0),
            "vulkan" => CreateVulkan(0),
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
