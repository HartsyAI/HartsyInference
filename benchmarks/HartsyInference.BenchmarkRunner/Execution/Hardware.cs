using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;
using HartsyInference.Engine;
using HartsyInference.BenchmarkRunner.Contracts;
using HartsyInference.BenchmarkRunner.Serialization;

namespace HartsyInference.BenchmarkRunner.Execution;
/// <summary>Actual backend identity and bounded, non-secret execution provenance.</summary>
public static class Hardware
{
    public static string MachineId()
    {
        string path = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "hartsy-bench", "identity");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        if (!File.Exists(path))
            File.WriteAllText(path, Guid.NewGuid().ToString("N"));
        return Hashes.Text("hartsy-bench-v1:" + File.ReadAllText(path));
    }

    public static DeviceRecord Describe(string selector, IBackend backend)
    {
        string name = "CPU", identity = MachineId(), driver = "runtime", hardwareKind = "cpu";
        long memory = 0;
        string capabilities = RuntimeInformation.ProcessArchitecture.ToString();
        if (backend is CudaBackend cuda)
        {
            hardwareKind = "gpu";
            name = cuda.Context.DeviceName;
            identity = Hashes.Text("hartsy-bench-v1:" + cuda.Context.QueryDeviceUuid());
            driver = cuda.Context.QueryDriverVersion().ToString(CultureInfo.InvariantCulture);
            memory = checked((long)cuda.Context.TotalMemory);
            capabilities = $"sm_{cuda.Context.ComputeCapabilityMajor}{cuda.Context.ComputeCapabilityMinor}";
        }
        else if (backend is VulkanBackend vulkan)
        {
            hardwareKind = vulkan.Vk.DeviceType is VkPhysicalDeviceType.DiscreteGpu or VkPhysicalDeviceType
                .IntegratedGpu or VkPhysicalDeviceType.VirtualGpu ? "gpu" : "cpu";
            name = vulkan.Vk.DeviceName;
            identity = Hashes.Text("hartsy-bench-v1:" + vulkan.Vk.DeviceUuid);
            driver = $"{vulkan.Vk.VendorId}:{vulkan.Vk.DriverVersion}";
            memory = checked((long)vulkan.Vk.TotalVramBytes);
            capabilities = $"vulkan-api:{vulkan.Vk.ApiVersion};device-type:{vulkan.Vk.DeviceType}";
        }

        return new DeviceRecord
        {
            Selector = selector,
            Name = name,
            Identity = identity,
            Driver = driver,
            MemoryBytes = memory,
            Capabilities = capabilities,
            HardwareKind = hardwareKind
        };
    }

    public static DeviceRecord Probe(string selector)
    {
        if (selector != "cpu" && !System.Text.RegularExpressions.Regex.IsMatch(selector, @"^(cuda|vulkan):[0-9]{1,2}$"))
            throw new ArgumentException("Use cpu, cuda:0, or vulkan:0 (an explicit ordinal is required).");
        using IBackend backend = BackendFactory.Create(selector);
        return Describe(selector, backend);
    }

    public static KnobProfile Defaults(out SortedDictionary<string, string> settings)
    {
        settings = new(StringComparer.Ordinal);
        KnobProfile profile = KnobProfile.Create("benchmark-defaults");
        foreach (object knob in KnobRegistry.All)
        {
            (string id, string? env, string type, object? value, KnobScope scope, KnobDomain domain, string summary) = KnobRegistry
                .Describe(knob);
            settings[id] = Convert.ToString(value, CultureInfo.InvariantCulture) ?? "<null>";
            profile = knob switch
            {
                Knob<bool> k => profile.With(k, k.Default),
                Knob<int> k => profile.With(k, k.Default),
                Knob<long> k => profile.With(k, k.Default),
                Knob<float> k => profile.With(k, k.Default),
                Knob<bool?> k => profile.With(k, k.Default),
                Knob<int?> k => profile.With(k, k.Default),
                Knob<long?> k => profile.With(k, k.Default),
                Knob<float?> k => profile.With(k, k.Default),
                Knob<string> k => profile.With(k, k.Default),
                _ => throw new InvalidDataException("Unsupported knob type: " + type),
            };
        }

        return profile;
    }

    public static SortedDictionary<string, string> NativeLibraries()
    {
        SortedDictionary<string, string> result = new(StringComparer.Ordinal);
        foreach (ProcessModule module in Process.GetCurrentProcess().Modules)
        {
            string name = Path.GetFileName(module.FileName);
            if (name.Contains("cuda", StringComparison.OrdinalIgnoreCase) || name.Contains("vulkan", StringComparison.OrdinalIgnoreCase)
                || name.Contains("cublas", StringComparison.OrdinalIgnoreCase) || name.Contains("cudnn", StringComparison.OrdinalIgnoreCase))
                result[name] = Hashes.FileHash(module.FileName);
        }

        return result;
    }

    public static EnvironmentRecord Capture(DeviceRecord device)
    {
        Defaults(out SortedDictionary<string, string> settings);
        SortedDictionary<string, string> binaries = new(StringComparer.Ordinal);
        foreach (string file in Directory.EnumerateFiles(AppContext.BaseDirectory, "*", SearchOption.AllDirectories).Where(p => p.EndsWith(
            ".dll") || p.EndsWith(".ptx") || p.EndsWith(".spv")))
            binaries[Path.GetRelativePath(AppContext.BaseDirectory, file).Replace('\\', '/')] = Hashes.FileHash(file);
        string revision = typeof(Hardware).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>().Single(a => a
            .Key == "BenchmarkRevision").Value!;
        foreach (string key in new[]
        {
            "CUDA_VISIBLE_DEVICES",
            "CUDA_DEVICE_ORDER",
            "CUDA_MODULE_LOADING",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "DOTNET_TieredPGO",
            "DOTNET_ReadyToRun",
            "DOTNET_GCHeapHardLimit",
            "DOTNET_gcServer",
            "VK_ICD_FILENAMES",
            "VK_DRIVER_FILES"
        }

        )
        {
            string? value = Environment.GetEnvironmentVariable(key);
            settings["environment." + key] = value is null ? "<unset>" : key.StartsWith("VK_", StringComparison.Ordinal)
                || key == "CUDA_VISIBLE_DEVICES" ? "sha256:" + Hashes.Text(value) : value;
        }

        return new EnvironmentRecord
        {
            MachineId = MachineId(),
            EngineRevision = revision,
            EngineVersion = typeof(InferenceEngine).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()
                ?.InformationalVersion ?? typeof(InferenceEngine).Assembly.GetName().Version!.ToString(),
            OperatingSystem = RuntimeInformation.OSDescription,
            Runtime = RuntimeInformation.FrameworkDescription,
            Architecture = RuntimeInformation.ProcessArchitecture.ToString(),
            CpuCount = Environment.ProcessorCount,
            Device = device,
            Binaries = binaries,
            Settings = settings
        };
    }
}
