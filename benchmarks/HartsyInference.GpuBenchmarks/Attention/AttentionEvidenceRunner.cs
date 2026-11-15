using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;

namespace HartsyInference.GpuBenchmarks.Attention;

/// <summary>Measures verified shared fixtures; reflection for optional telemetry stays outside timed operations.</summary>
internal static class AttentionEvidenceRunner
{
    public static int Run(string[] args)
    {
        Dictionary<string, string> options = new(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i += 2)
        {
            if (i + 1 >= args.Length || !args[i].StartsWith("--", StringComparison.Ordinal))
                throw new ArgumentException("Evidence arguments must be --name value pairs.");
            options.Add(args[i][2..], args[i + 1]);
        }
        string boundary = options.GetValueOrDefault("boundary", "public");
        if (boundary is not ("public" or "resident" or "observe")) throw new ArgumentException("Unknown timing boundary.");
        int trials = int.Parse(options.GetValueOrDefault("trials", "1"), System.Globalization.CultureInfo.InvariantCulture);
        if (trials <= 0) throw new ArgumentOutOfRangeException(nameof(args));
        using AttentionFixture fixture = AttentionFixture.Load(options["fixture"]);
        if (options.GetValueOrDefault("mandatory", "false") == "true" && !fixture.AcceptanceReviewed)
            throw new InvalidOperationException("Unreviewed acceptance cannot certify a mandatory case.");
        AttentionArm arm = AttentionArms.Get(options.GetValueOrDefault("arm", "def-f16"));
        AttentionArms.ClearAll();
        arm.Apply();
        using IDisposable settingsScope = HartsyInference.Benchmarks.Shared.BenchmarkKnobProfiles
            .Capture(true, out SortedDictionary<string, string> settings).Push();
        using BenchmarkFixture hardware = new();
        IBackend backend = hardware.Backend;
        arm.Configure?.Invoke(backend);
        AttentionCall call = fixture.Call;
        Action operation = () => backend.Attention(in call);
        Func<string?> route = BranchReader(backend);
        AttentionOracle oracle = fixture.Oracle();
        IDisposable? evidence = null;
        string? observationPath = options.GetValueOrDefault("observe");
        try
        {
            if (observationPath is not null)
            {
                Type? observer = typeof(AttentionCall).Assembly.GetType("HartsyInference.Core.Backends.Attention.AttentionEvidence");
                MethodInfo begin = observer?.GetMethod("Begin")
                    ?? throw new NotSupportedException("Pristine binaries have no PR 01 telemetry; use disabled timing.");
                evidence = (IDisposable)begin.Invoke(null, [65536])!;
            }
            for (int i = 0; i < 10; i++) { operation(); backend.Sync(); }
            string? initialRoute = route();
            string expectedRoute = options.GetValueOrDefault("expect-branch", initialRoute ?? "");
            if (string.IsNullOrEmpty(expectedRoute) || initialRoute != expectedRoute)
                throw new InvalidOperationException("Executed branch does not match the requested evidence cell.");
            double[] milliseconds = new double[trials];
            using DeviceTimer timer = new(backend, boundary == "resident");
            for (int i = 0; i < trials; i++)
            {
                backend.Sync();
                if (boundary == "resident") milliseconds[i] = timer.Measure(operation) / 32;
                else
                {
                    long start = Stopwatch.GetTimestamp();
                    operation();
                    backend.Sync();
                    milliseconds[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                }
                if (route() != expectedRoute)
                    throw new InvalidOperationException("Instrumentation or memory state changed the executed branch.");
            }
            (double rel, double cosine, double maxAbs, int rows) = oracle.Compare(fixture.OutputAt, 64);
            bool quality = double.IsFinite(rel) && rel <= fixture.RelativeTolerance;
            double sumAbsolute = 0;
            long elements = 0;
            foreach ((int b, int h, int row) in oracle.Sample(64))
            {
                double[] expected = oracle.Row(b, h, row);
                for (int c = 0; c < expected.Length; c++)
                {
                    sumAbsolute += Math.Abs(expected[c] - fixture.OutputAt(b, h, row, c));
                    elements++;
                }
            }
            double meanAbsolute = sumAbsolute / elements;
            if (fixture.MeanAbsoluteTolerance is { } meanLimit) quality &= meanAbsolute <= meanLimit;
            if (evidence is not null)
            {
                evidence.Dispose();
                using FileStream capture = new(observationPath!, FileMode.CreateNew);
                evidence.GetType().GetMethod("WriteJson")!.Invoke(evidence, [capture]);
            }
            using FileStream result = new(options["out"], FileMode.CreateNew);
            using Utf8JsonWriter writer = new(result, new JsonWriterOptions { Indented = true });
            writer.WriteStartObject();
            writer.WriteNumber("schema", 1);
            writer.WriteString("case_id", fixture.CaseId);
            writer.WriteString("fixture_sha256", fixture.ManifestHash);
            writer.WriteString("backend", hardware.Kind.ToString().ToLowerInvariant());
            writer.WriteString("device", backend.Capabilities.DeviceName);
            writer.WriteString("arm", arm.Name);
            writer.WriteString("knobs", AttentionArms.Provenance());
            writer.WriteStartObject("resolved_settings");
            foreach ((string key, string value) in settings) writer.WriteString(key, value);
            writer.WriteEndObject();
            writer.WriteString("branch", expectedRoute);
            writer.WriteString("boundary", boundary);
            writer.WriteString("telemetry", evidence is null ? "disabled" : "enabled");
            writer.WriteString("core_sha256", Hash(typeof(AttentionCall).Assembly.Location));
            writer.WriteString("backend_sha256", Hash(backend.GetType().Assembly.Location));
            writer.WriteString("output_sha256", fixture.OutputHash());
            writer.WriteString("cohort", options.GetValueOrDefault("cohort", "unregistered"));
            writer.WriteBoolean("quality_passed", quality);
            writer.WriteBoolean("acceptance_reviewed", fixture.AcceptanceReviewed);
            writer.WriteNumber("oracle_rows", rows);
            if (double.IsFinite(rel)) writer.WriteNumber("rel_l2", rel); else writer.WriteNull("rel_l2");
            if (double.IsFinite(cosine)) writer.WriteNumber("cosine", cosine); else writer.WriteNull("cosine");
            if (double.IsFinite(maxAbs)) writer.WriteNumber("max_abs", maxAbs); else writer.WriteNull("max_abs");
            if (double.IsFinite(meanAbsolute)) writer.WriteNumber("mean_abs", meanAbsolute); else writer.WriteNull("mean_abs");
            writer.WriteNumber("warmup", 10);
            writer.WriteNumber("iterations", boundary == "resident" ? 32 : 1);
            writer.WriteNumber("empty_device_region_ms", timer.EmptyMilliseconds);
            if (timer.TimestampPeriod > 0) writer.WriteNumber("timestamp_period_ns", timer.TimestampPeriod);
            if (timer.TimestampBits > 0) writer.WriteNumber("timestamp_valid_bits", timer.TimestampBits);
            WriteHashes(writer, "ptx_sha256", Path.Combine(AppContext.BaseDirectory, "Ptx"));
            WriteHashes(writer, "spirv_sha256", Path.Combine(AppContext.BaseDirectory, "Spirv"));
            writer.WriteStartObject("native_libraries");
            foreach (ProcessModule module in Process.GetCurrentProcess().Modules)
            {
                string name = module.ModuleName.ToLowerInvariant();
                if (name.StartsWith("hartsyinference.", StringComparison.Ordinal)) continue;
                if (name.Contains("cublas") || name.Contains("cudnn") || name.Contains("nvcuda") || name.Contains("vulkan"))
                    writer.WriteString(module.FileName, Hash(module.FileName));
            }
            writer.WriteEndObject();
            writer.WriteStartArray("milliseconds");
            foreach (double value in milliseconds) writer.WriteNumberValue(value);
            writer.WriteEndArray();
            writer.WriteEndObject();
            return quality ? 0 : 1;
        }
        finally
        {
            evidence?.Dispose();
            AttentionArms.ClearAll();
        }
    }

    private static string Hash(string path)
    {
        using FileStream input = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(input)).ToLowerInvariant();
    }

    private static void WriteHashes(Utf8JsonWriter writer, string name, string directory)
    {
        writer.WriteStartObject(name);
        if (Directory.Exists(directory))
            foreach (string path in Directory.EnumerateFiles(directory).Order(StringComparer.Ordinal))
                writer.WriteString(Path.GetFileName(path), Hash(path));
        writer.WriteEndObject();
    }

    private static Func<string?> BranchReader(IBackend backend)
    {
        MethodInfo getter = backend.GetType().GetProperty("LastAttentionRoute", BindingFlags.Static | BindingFlags.NonPublic)?.GetMethod
            ?? throw new NotSupportedException("This backend has no baseline executed-branch reader.");
        return getter.CreateDelegate<Func<string?>>();
    }

    private sealed class DeviceTimer : IDisposable
    {
        private readonly nint _stream;
        private nint _start;
        private nint _end;
        private readonly Func<int, Action, double>? _vulkan;

        public double EmptyMilliseconds { get; }
        public float TimestampPeriod { get; }
        public uint TimestampBits { get; }

        public DeviceTimer(IBackend backend, bool enabled)
        {
            if (!enabled) return;
            if (backend is CudaBackend)
            {
                object stream = backend.GetType().GetField("_stream", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(backend)!;
                _stream = (nint)stream.GetType().GetProperty("Handle")!.GetValue(stream)!;
                CudaDriverApi.cuEventCreate(out _start, 0).ThrowOnError();
                try { CudaDriverApi.cuEventCreate(out _end, 0).ThrowOnError(); }
                catch { Release(false); throw; }
                CudaDriverApi.cuEventRecord(_start, _stream).ThrowOnError();
                CudaDriverApi.cuEventRecord(_end, _stream).ThrowOnError();
                CudaDriverApi.cuEventSynchronize(_end).ThrowOnError();
                CudaDriverApi.cuEventElapsedTime(out float empty, _start, _end).ThrowOnError();
                EmptyMilliseconds = empty;
            }
            else if (backend is VulkanBackend vk)
            {
                TimestampPeriod = vk.Vk.TimestampPeriod;
                TimestampBits = QueryTimestampBits(vk);
                if (!float.IsFinite(TimestampPeriod) || TimestampPeriod <= 0 || TimestampBits != 64)
                    throw new NotSupportedException("Baseline timestamp reader requires a valid period and 64 valid bits.");
                MethodInfo method = backend.GetType().GetMethod("MeasureGpuTimeMs", BindingFlags.Instance | BindingFlags.NonPublic)!;
                _vulkan = method.CreateDelegate<Func<int, Action, double>>(backend);
                EmptyMilliseconds = _vulkan(0, static () => { });
            }
            else throw new NotSupportedException("Device timestamps are unavailable.");
        }

        public double Measure(Action operation)
        {
            if (_vulkan is not null) return _vulkan(32, operation);
            CudaDriverApi.cuEventRecord(_start, _stream).ThrowOnError();
            for (int i = 0; i < 32; i++) operation();
            CudaDriverApi.cuEventRecord(_end, _stream).ThrowOnError();
            CudaDriverApi.cuEventSynchronize(_end).ThrowOnError();
            CudaDriverApi.cuEventElapsedTime(out float elapsed, _start, _end).ThrowOnError();
            return elapsed;
        }

        public void Dispose()
        {
            Release(true);
            GC.SuppressFinalize(this);
        }

        ~DeviceTimer() => Release(false);

        private void Release(bool check)
        {
            nint end = Interlocked.Exchange(ref _end, 0);
            nint start = Interlocked.Exchange(ref _start, 0);
            // Finalization may follow context teardown; release is best effort there.
            int endStatus = end == 0 ? 0 : CudaDriverApi.cuEventDestroy(end);
            int startStatus = start == 0 ? 0 : CudaDriverApi.cuEventDestroy(start);
            if (check)
            {
                endStatus.ThrowOnError();
                startStatus.ThrowOnError();
            }
        }

        private static unsafe uint QueryTimestampBits(VulkanBackend backend)
        {
            VulkanDevice device = (VulkanDevice)typeof(VulkanBackend)
                .GetField("_vkDevice", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(backend)!;
            Type api = typeof(VulkanBackend).Assembly.GetType("HartsyInference.Vulkan.VulkanApi")!;
            QueueQuery query = api.GetMethod("vkGetPhysicalDeviceQueueFamilyProperties", BindingFlags.Static | BindingFlags.NonPublic)!
                .CreateDelegate<QueueQuery>();
            uint count = 0;
            query(device.PhysicalDevice, ref count, 0);
            if (count == 0 || count > 1024) throw new NotSupportedException("Invalid Vulkan queue-family count.");
            nint storage = Marshal.AllocHGlobal(checked((int)count * sizeof(VkQueueFamilyProperties)));
            try
            {
                query(device.PhysicalDevice, ref count, storage);
                if (backend.Vk.ComputeQueueFamilyIndex >= count) throw new NotSupportedException("Compute queue disappeared.");
                return ((VkQueueFamilyProperties*)storage)[backend.Vk.ComputeQueueFamilyIndex].timestampValidBits;
            }
            finally { Marshal.FreeHGlobal(storage); }
        }

        private delegate void QueueQuery(nint physicalDevice, ref uint count, nint properties);
    }
}
