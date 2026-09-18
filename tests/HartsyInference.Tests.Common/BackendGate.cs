using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;

namespace HartsyInference.Tests.Common;

/// <summary>Opens a backend for a test, or explains why it cannot. The backend equivalent of
/// <see cref="RealWeightGate"/>, and deliberately the same shape:
/// <c>if (!BackendGate.TryOpen("vulkan", _output.WriteLine, out IBackend? gpu)) return;</c>
///
/// <para>Two things this fixes about how GPU tests are written today. Each test file carries its own private
/// availability check — some thirty copies of a <c>PtxDir()</c> helper on the CUDA side, a <c>VulkanAvailable()</c>
/// on the other — so the kernel-directory rules and the skip policy are decided separately in every file. And every
/// one of those checks returns early on a miss, which xunit records as a <b>PASS</b>: a machine with no GPU reports
/// a fully green GPU suite. <see cref="RequireEnvVar"/> turns that into a hard failure, so a verification run cannot
/// come back green because the hardware was absent.</para></summary>
public static class BackendGate
{
    /// <summary>Set to 1 to make an unavailable backend fail the test instead of skipping it.</summary>
    public const string RequireEnvVar = "HARTSY_REQUIRE_BACKENDS";

    /// <summary>Set to 1 to accept a software Vulkan device (llvmpipe/lavapipe). Off by default: it is useful for
    /// checking small-subgroup correctness and worthless as evidence about hardware, so it must be asked for.</summary>
    public const string AllowSoftwareEnvVar = "HARTSY_ALLOW_SOFTWARE_GPU";

    /// <summary>Backend names these helpers understand.</summary>
    public static IReadOnlyList<string> Kinds { get; } = ["cpu", "cuda", "vulkan"];

    /// <summary>Every GPU backend, for a theory that should run on each. Names only — availability is decided when
    /// the test opens one, so a machine without a given GPU reports per-test rather than losing the rows entirely.</summary>
    public static IEnumerable<object[]> GpuKinds => [["cuda"], ["vulkan"]];

    /// <summary>Every backend including the CPU one, for a theory comparing all of them.</summary>
    public static IEnumerable<object[]> AllKinds => [["cpu"], ["cuda"], ["vulkan"]];

    private static readonly Dictionary<string, string?> _unavailable = new(StringComparer.OrdinalIgnoreCase);
    private static readonly object _probeLock = new();

    /// <summary>Opens <paramref name="kind"/>, or reports why not. Returns false after logging a SKIPPED line when
    /// the backend is unavailable — unless <see cref="RequireEnvVar"/> is set, in which case it throws.</summary>
    /// <remarks>The caller owns the returned backend and should <c>using</c> it. A fresh instance per test rather
    /// than a shared one: these tests routinely tear a backend down to check that teardown itself is correct, and
    /// three of the bugs found migrating Vulkan's residency cache were only reachable that way.</remarks>
    public static bool TryOpen(string kind, Action<string> log, out IBackend? backend)
    {
        ArgumentNullException.ThrowIfNull(log);
        backend = null;
        string? reason = UnavailableReason(kind);
        if (reason is not null)
        {
            if (Environment.GetEnvironmentVariable(RequireEnvVar) == "1")
            {
                throw new InvalidOperationException(
                    $"{RequireEnvVar}=1 but the '{kind}' backend is unavailable: {reason}");
            }
            log($"SKIPPED: {kind} backend unavailable — {reason}");
            return false;
        }
        backend = Create(kind);
        return true;
    }

    /// <summary>Why <paramref name="kind"/> cannot be opened here, or null when it can. Probed once per kind.</summary>
    public static string? UnavailableReason(string kind)
    {
        lock (_probeLock)
        {
            if (_unavailable.TryGetValue(kind, out string? cached))
            {
                return cached;
            }
            string? reason = Probe(kind);
            _unavailable[kind] = reason;
            return reason;
        }
    }

    private static string? Probe(string kind)
    {
        switch (kind.ToLowerInvariant())
        {
            case "cpu":
                return null;

            case "cuda":
                if (!CudaContext.IsAvailable())
                {
                    return CudaContext.LastUnavailableReason ?? "no CUDA driver or device";
                }
                if (CudaContext.GetDeviceCount() == 0)
                {
                    return "no CUDA devices";
                }
                return KernelDir("Ptx", "HartsyInference.Cuda") is null
                    ? "no compiled PTX directory beside the tests or in the repo"
                    : null;

            case "vulkan":
                try
                {
                    using VulkanInstance instance = new();
                    if (instance.EnumeratePhysicalDevices().Length == 0)
                    {
                        return "no Vulkan physical devices";
                    }
                }
                catch (Exception ex)
                {
                    return $"Vulkan loader failed: {ex.GetType().Name}: {ex.Message}";
                }
                if (KernelDir("Spirv", "HartsyInference.Vulkan") is null)
                {
                    return "no compiled SPIR-V directory beside the tests or in the repo";
                }
                // A software rasterizer reports the host's silicon vendor like any other device, so it has to be
                // excluded explicitly or it silently stands in for hardware in anything measured.
                if (Environment.GetEnvironmentVariable(AllowSoftwareEnvVar) != "1")
                {
                    using VulkanBackend probe = new(0, KernelDir("Spirv", "HartsyInference.Vulkan"));
                    if (probe.Capabilities.Vendor == GpuVendor.Software)
                    {
                        return $"only a software Vulkan device is present ({probe.Capabilities.DeviceName}); "
                            + $"set {AllowSoftwareEnvVar}=1 to test against it";
                    }
                }
                return null;

            default:
                return $"unknown backend kind '{kind}'";
        }
    }

    private static IBackend Create(string kind) => kind.ToLowerInvariant() switch
    {
        "cpu" => new CpuBackend(),
        "cuda" => new CudaBackend(0, KernelDir("Ptx", "HartsyInference.Cuda")),
        "vulkan" => new VulkanBackend(0, KernelDir("Spirv", "HartsyInference.Vulkan")),
        _ => throw new ArgumentException($"Unknown backend kind '{kind}'.", nameof(kind)),
    };

    /// <summary>The compiled-kernel directory: beside the test binaries when the build copied it there, otherwise
    /// the one in the source tree. Null when neither exists.</summary>
    /// <remarks>This rule was copy-pasted into roughly thirty test files, each free to get it subtly wrong or to
    /// stop matching when the build layout changed. It belongs in exactly one place.</remarks>
    public static string? KernelDir(string subdirectory, string projectName)
    {
        string beside = Path.Combine(AppContext.BaseDirectory, subdirectory);
        if (Directory.Exists(beside))
        {
            return beside;
        }
        string inRepo = Path.Combine(RepoRoot.Path, "src", projectName, subdirectory);
        return Directory.Exists(inRepo) ? inRepo : null;
    }
}
