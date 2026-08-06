using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>First REAL-WEIGHT concurrent exercise of the same-GPU dual-backend feature: two engines on the SAME
/// ordinal (0, the 4090) hold two different small models and generate CONCURRENTLY under the
/// <c>HARTSY_SAME_GPU_CONCURRENT=1</c> opt-in. Correctness bar is bit-identity against each engine's own
/// serialized run — same engine, same seed, same device, so any divergence is cross-backend state bleed, the
/// engine-level counterpart of what <c>MultiBackendIsolationTests</c> proves at the backend/unit level.
/// IMPORTANT: <c>HARTSY_SAME_GPU_CONCURRENT</c> is read ONCE per process (<c>DeviceGate</c> static init), so the
/// fact sets it via <c>Environment.SetEnvironmentVariable</c> as its FIRST LINE and the campaign must run this
/// class filter-isolated in its own test-host process.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class SameGpuConcurrentRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public SameGpuConcurrentRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task TwoEnginesOneGpu_ConcurrentGenerations_BitIdenticalToSerialized()
    {
        Environment.SetEnvironmentVariable("HARTSY_SAME_GPU_CONCURRENT", "1");

        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string sdxlCheckpoint = TestPaths.Sdxl.SingleFile;
        string schnellCheckpoint = TestPaths.Flux.Schnell;
        if (!RealWeightGate.Require(_output.WriteLine, sdxlCheckpoint, schnellCheckpoint)) return;
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(SameGpuConcurrentRealWeightTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        double free0 = ProbeFreeGb();
        _output.WriteLine($"Free VRAM — ordinal 0: {free0:F2} GB.");
        if (free0 < 19.0)
        {
            _output.WriteLine("SKIPPED: SDXL (~7 GB) and the schnell fp8 DiT (~12 GB) must co-reside on ordinal 0.");
            return;
        }

        ModelSpec sdxlSpec = ModelResolver.Resolve("sdxl", sdxlCheckpoint, Modality.Image);
        if (sdxlSpec.LocalPath is null) { _output.WriteLine("SKIPPED: sdxl not resolvable with the explicit path."); return; }
        ModelSpec schnellSpec = ModelResolver.Resolve("flux1", schnellCheckpoint, Modality.Image);
        if (schnellSpec.LocalPath is null) { _output.WriteLine("SKIPPED: flux1 not resolvable with the explicit path."); return; }

        ImageRequest sdxlRequest = new ImageRequest
        {
            Prompt = "a photograph of an astronaut riding a horse",
            Width = 512,
            Height = 512,
            Steps = 8,
            CfgScale = 6.0f,
            Seed = 42,
        };
        // Schnell is distilled: 4 steps, no CFG.
        ImageRequest schnellRequest = new ImageRequest
        {
            Prompt = "a photograph of an astronaut riding a horse",
            Width = 512,
            Height = 512,
            Steps = 4,
            CfgScale = 1.0f,
            Seed = 42,
        };

        // Explicit disposal (not using-declarations): a Dispose that throws during exception unwinding would
        // REPLACE the original failure — log both, surface the original.
        InferenceEngine engineA = new InferenceEngine("cuda", 0);
        InferenceEngine engineB = new InferenceEngine("cuda", 0);
        try
        {

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine("[1/2] SERIAL reference pass (loads both models, keeps them resident)...");
        ImageResult sdxlSerial = await engineA.Images.GenerateAsync(sdxlSpec, sdxlRequest);
        ImageResult schnellSerial = await engineB.Images.GenerateAsync(schnellSpec, schnellRequest);
        double serialSeconds = sw.Elapsed.TotalSeconds;
        _output.WriteLine($"  serial pair: sdxl {sdxlSerial.Width}x{sdxlSerial.Height}, schnell {schnellSerial.Width}x{schnellSerial.Height}, {serialSeconds:F1}s");
        AssertCoherent(sdxlSerial.Rgb, "sdxl serial");
        AssertCoherent(schnellSerial.Rgb, "schnell serial");

        sw.Restart();
        _output.WriteLine("[2/2] CONCURRENT pass (same two engines, weights already resident)...");
        Task<ImageResult> sdxlTask = engineA.Images.GenerateAsync(sdxlSpec, sdxlRequest);
        Task<ImageResult> schnellTask = engineB.Images.GenerateAsync(schnellSpec, schnellRequest);
        await Task.WhenAll(sdxlTask, schnellTask);
        double concurrentSeconds = sw.Elapsed.TotalSeconds;
        ImageResult sdxlConcurrent = await sdxlTask;
        ImageResult schnellConcurrent = await schnellTask;
        _output.WriteLine($"  concurrent pair: {concurrentSeconds:F1}s (serial pair was {serialSeconds:F1}s incl. loads)");

        Assert.True(sdxlSerial.Rgb.AsSpan().SequenceEqual(sdxlConcurrent.Rgb),
            "SDXL concurrent output differs from its own serialized run on the same engine/seed/device — " +
            "bit-identity is the correct bar here, so any divergence indicates cross-backend state bleed " +
            "(see MultiBackendIsolationTests for the unit-level sibling).");
        Assert.True(schnellSerial.Rgb.AsSpan().SequenceEqual(schnellConcurrent.Rgb),
            "Flux-schnell concurrent output differs from its own serialized run on the same engine/seed/device — " +
            "bit-identity is the correct bar here, so any divergence indicates cross-backend state bleed " +
            "(see MultiBackendIsolationTests for the unit-level sibling).");
        }
        catch (Exception ex)
        {
            _output.WriteLine($"ORIGINAL failure (before any Dispose): {ex}");
            throw;
        }
        finally
        {
            try { engineB.Dispose(); }
            catch (Exception ex) { _output.WriteLine($"engineB.Dispose threw: {ex}"); }
            try { engineA.Dispose(); }
            catch (Exception ex) { _output.WriteLine($"engineA.Dispose threw: {ex}"); }
        }
    }

    private static void AssertCoherent(byte[] rgb, string label)
    {
        int nonZero = rgb.Count(b => b != 0), nonFF = rgb.Count(b => b != 255);
        Assert.True(nonZero > rgb.Length * 0.1, $"{label}: image is all black");
        Assert.True(nonFF > rgb.Length * 0.1, $"{label}: image is all white");
    }

    private static double ProbeFreeGb()
    {
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(SameGpuConcurrentRealWeightTests).Assembly.Location)!, "Ptx");
        using CudaBackend probe0 = new(deviceOrdinal: 0, ptxDir: ptxDir);
        (nuint free0, _) = probe0.Context.GetMemoryInfo();
        return free0 / (1024.0 * 1024.0 * 1024.0);
    }
}
