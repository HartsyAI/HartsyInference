using System.Diagnostics;
using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Cuda.Profiling;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Times the full SANA-WM DiT on the recorded step-0 inputs (CFG batch 2): one warmup forward, timed forwards, then
/// one forward under the stage profiler and one under the synced per-op profiler. Writes <c>stages.tsv</c>,
/// <c>ops_cpu.txt</c> and <c>ops_gpu.txt</c> to <c>SANAWM_PROFILE_OUT</c> when set. Benchmark lane:
/// <c>Category=GpuBenchmark&amp;FullyQualifiedName~SanaWmDitPerf</c>; needs a CUDA device, <c>HARTSYINFERENCE_MODELS_DIR</c>
/// and <c>SANAWM_REFERENCE_RUN_DIR</c>.</summary>
[Trait("Category", "GpuBenchmark")]
public sealed class SanaWmDitPerfGpuTests
{
    private const double CosineTarget = 0.999;
    private readonly ITestOutputHelper _output;

    public SanaWmDitPerfGpuTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [SanaWmBlockReferenceFact]
    public void FullDit_StepTimeAndStageProfile()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        string? outDirectory = Environment.GetEnvironmentVariable("SANAWM_PROFILE_OUT");
        int timedRuns = int.TryParse(Environment.GetEnvironmentVariable("SANAWM_PROFILE_RUNS"), out int runs) ? runs : 2;
        string runDirectory = SanaWmBlockReferenceFactAttribute.ReferenceDirectory();
        using SanaWmCheckpoint checkpoint = SanaWmCheckpointConverter.Load(SanaWmRealWeightsFactAttribute.ResolveDitPath());
        using CudaBackend backend = new(0, PtxDirectory());
        IBackend device = backend;
        using SanaWmDit dit = new(backend, checkpoint);
        using SafeTensorsLoader sampler = Load(Path.Combine(runDirectory, "sampler.safetensors"));
        using SafeTensorsLoader text = Load(Path.Combine(runDirectory, "text.safetensors"));
        using SafeTensorsLoader camera = Load(Path.Combine(runDirectory, "camera.safetensors"));
        List<Tensor> owned = [];
        try
        {
            Tensor x = Host32(sampler.GetTensor("dit_in_latents_step0"), owned);
            Tensor caption = Host32(text.GetTensor("dit_y_step0"), owned);
            Tensor mask = MaskAsFloat(text.GetTensor("dit_mask_step0"), owned);
            Tensor chunk = Host32(camera.GetTensor("dit_chunk_plucker_step0"), owned);
            float[] timesteps = Host32(sampler.GetTensor("dit_timestep_step0"), owned).AsReadOnlySpan<float>().ToArray();
            float[] cameraConditions = Host32(camera.GetTensor("dit_camera_conditions_step0"), owned).AsReadOnlySpan<float>().ToArray();
            Tensor expected = Host32(sampler.GetTensor("dit_out_raw_step0"), owned);

            Tensor Run() => dit.Forward(x, timesteps, caption, mask, cameraConditions, chunk);

            Stopwatch clock = Stopwatch.StartNew();
            using (Tensor warm = Run())
            {
                device.OffloadActivation(warm);
                clock.Stop();
                double cosine = Cosine(warm.AsReadOnlySpan<float>(), expected.AsReadOnlySpan<float>());
                _output.WriteLine($"warmup forward: {clock.Elapsed.TotalSeconds:F3} s, cosine {cosine:F6}");
                Assert.True(cosine >= CosineTarget, $"DiT cosine {cosine:F6} is below {CosineTarget}.");
            }

            for (int i = 0; i < timedRuns; i++)
            {
                Process self = Process.GetCurrentProcess();
                TimeSpan cpuBefore = self.TotalProcessorTime;
                clock.Restart();
                using Tensor velocity = Run();
                double submitted = clock.Elapsed.TotalSeconds;
                device.Sync();
                double total = clock.Elapsed.TotalSeconds;
                self.Refresh();
                double cpu = (self.TotalProcessorTime - cpuBefore).TotalSeconds;
                _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"timed forward {i}: {total:F3} s total, {submitted:F3} s until Forward returned, {cpu:F3} s process CPU"));
            }

            try
            {
                KnobStore.Set(EngineKnobs.Profile, true);
                KnobStore.Set(EngineKnobs.StageProfile, true);
                StageProfiler.Reset();
                NvtxRange.ResetProfile();
                clock.Restart();
                using (Tensor staged = Run())
                    device.Sync();
                _output.WriteLine($"stage-profiled forward: {clock.Elapsed.TotalSeconds:F3} s");
                string stages = StageProfiler.Report();
                _output.WriteLine(stages);
                if (outDirectory is not null)
                {
                    Directory.CreateDirectory(outDirectory);
                    File.WriteAllText(Path.Combine(outDirectory, "stages.tsv"), stages);
                    NvtxRange.DumpProfile(Path.Combine(outDirectory, "ops_cpu.txt"));
                }

                KnobStore.Set(EngineKnobs.ProfileSync, true);
                KnobStore.Set(EngineKnobs.ProfileFine, true);
                StageProfiler.Reset();
                NvtxRange.ResetProfile();
                clock.Restart();
                using (Tensor synced = Run())
                    device.Sync();
                _output.WriteLine($"op-synced forward: {clock.Elapsed.TotalSeconds:F3} s");
                if (outDirectory is not null)
                    NvtxRange.DumpProfile(Path.Combine(outDirectory, "ops_gpu.txt"));
            }
            finally
            {
                KnobStore.Clear(EngineKnobs.ProfileFine);
                KnobStore.Clear(EngineKnobs.ProfileSync);
                KnobStore.Clear(EngineKnobs.Profile);
                KnobStore.Clear(EngineKnobs.StageProfile);
            }
        }
        finally
        {
            SanaWmTensors.DisposeAll(owned);
        }
    }

    private static SafeTensorsLoader Load(string path)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        return loader;
    }

    private static Tensor Host32(Tensor source, List<Tensor> owned)
    {
        if (source.DType == DType.F32)
            return source;
        Tensor converted = source.CastTo(DType.F32);
        owned.Add(converted);
        return converted;
    }

    private static Tensor MaskAsFloat(Tensor mask, List<Tensor> owned)
    {
        Tensor converted = new(mask.Shape, DType.F32, DeviceKind.Cpu);
        ReadOnlySpan<long> source = mask.AsReadOnlySpan<long>();
        Span<float> destination = converted.AsSpan<float>();
        for (int i = 0; i < destination.Length; i++)
            destination[i] = source[i];
        owned.Add(converted);
        return converted;
    }

    private static double Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            normA += (double)a[i] * a[i];
            normB += (double)b[i] * b[i];
        }

        return dot / Math.Sqrt(Math.Max(normA * normB, double.Epsilon));
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate))
                return candidate;
        }

        throw new DirectoryNotFoundException("HartsyInference.Cuda/Ptx was not found above the test binary.");
    }
}
