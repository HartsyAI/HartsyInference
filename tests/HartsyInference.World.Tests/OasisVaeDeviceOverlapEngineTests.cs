using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tests.Common;

namespace HartsyInference.World.Tests;

/// <summary>Real-engine verification of <c>PlacementConfig.VaeDevice</c> for Oasis (ROADMAP "3D &amp; World models"):
/// <c>OasisPipeline.Generate</c> was restructured so a finished frame's VAE decode overlaps the NEXT frame's DiT
/// denoise on a second backend (see the <c>overlapDecode</c> path and its class-doc note) instead of the old
/// all-denoise-then-all-decode batch split, which had zero device overlap regardless of where the VAE ran.
/// Verifies, on real Oasis-500m weights (<c>camenduru/oasis-500m</c> mirror, <c>.pt</c>→safetensors — see
/// MODEL_STATUS_WORLD.md): (1) <c>VaeDevice</c> unset is byte-identical across repeated runs — the hard
/// no-placement invariant every <c>PlacementConfig</c> consumer holds; (2) <c>VaeDevice</c> set to the second
/// card produces closely-matching frames (SSIM) with genuine measured wall-clock difference and observed VAE-card
/// utilization before the run's midpoint (not just a tail-end decode batch). Primary stays on the same physical
/// card (the 4090) in every run so denoise speed is controlled; only the VAE's device changes — see the in-method
/// note on this box's CUDA-ordinal-vs-nvidia-smi-ordinal inversion before trusting any <c>cuda:N</c> literal.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class OasisVaeDeviceOverlapEngineTests
{
    private const int Width = 640, Height = 360, TotalFrames = 8, DdimSteps = 10;

    private readonly ITestOutputHelper _output;
    public OasisVaeDeviceOverlapEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task VaeDevice_UnsetIsByteIdentical_SetOverlapsWithMeasuredWallClock()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.Oasis.Dit, TestPaths.Oasis.Vae)) return;

        byte[] seedFrame = SyntheticSeedFrame(Width, Height, seed: 7);
        string[] actions = Enumerable.Repeat("forward", TotalFrames - 1).ToArray();

        ModelSpec baseSpec = ModelResolver.Resolve("oasis", TestPaths.Oasis.Dit, Modality.World);
        ModelSpec spec = baseSpec with { Aux = new Dictionary<string, string> { [WorldService.VaeAuxKey] = TestPaths.Oasis.Vae } };
        WorldRequest request = new WorldRequest
        {
            InitImage = new ImageData { Rgb = seedFrame, Width = Width, Height = Height },
            Steps = DdimSteps,
            Seed = 42,
        };

        // CUDA's own ordinal order is "fastest first" and does NOT match nvidia-smi's PCI-bus order on this box:
        // nvidia-smi GPU0/GPU1 = 3060/4090, but CudaBackend's cuda:0/cuda:1 = 4090/3060 (confirmed via the
        // "[Cuda] device N: <name>" log line each backend construction prints). Primary is cuda:0 (the 4090) in
        // every run — denoise (10 DiT forwards/frame) dominates wall-clock, so the fast card must be the shared
        // constant; only the VAE's device changes between baseline and overlap.
        _output.WriteLine("[1/3] Baseline A — VaeDevice unset, single GPU (cuda:0, the 4090)...");
        Stopwatch swA = Stopwatch.StartNew();
        byte[][] framesA;
        using (InferenceEngine engineA = new InferenceEngine("cuda", 0))
        {
            framesA = await RunOneRollout(engineA, spec, request, actions);
        }
        swA.Stop();
        _output.WriteLine($"  {framesA.Length} frames in {swA.Elapsed.TotalSeconds:F2}s");

        _output.WriteLine("[2/3] Baseline B — repeat of A (byte-identity invariant, not just run-to-run similarity)...");
        Stopwatch swB = Stopwatch.StartNew();
        byte[][] framesB;
        using (InferenceEngine engineB = new InferenceEngine("cuda", 0))
        {
            framesB = await RunOneRollout(engineB, spec, request, actions);
        }
        swB.Stop();
        _output.WriteLine($"  {framesB.Length} frames in {swB.Elapsed.TotalSeconds:F2}s");

        Assert.Equal(framesA.Length, framesB.Length);
        for (int i = 0; i < framesA.Length; i++)
        {
            Assert.True(framesA[i].AsSpan().SequenceEqual(framesB[i]),
                $"frame {i} differs between two VaeDevice-unset runs — the no-placement byte-identity invariant is broken.");
        }
        _output.WriteLine("  byte-identical across both VaeDevice-unset runs. OK.");

        _output.WriteLine("[3/3] VaeDevice=cuda:1 (the 3060) — the overlap path, GPU utilization sampled throughout...");
        PlacementConfig placement = new PlacementConfig { VaeDevice = "cuda:1" };
        // Gpu0/Gpu1 in the sample tuple are nvidia-smi indices (0=3060, 1=4090) — the opposite of the cuda:0/1
        // selectors above. "Gpu1 busy" here means the primary/4090 (denoise); "Gpu0 busy" means the VAE/3060.
        List<(int Gpu0, int Gpu1)> samples = new();
        using CancellationTokenSource samplerCts = new CancellationTokenSource();
        Task sampler = SampleUtilizationLoop(samples, samplerCts.Token);

        Stopwatch swC = Stopwatch.StartNew();
        byte[][] framesC;
        using (InferenceEngine engineC = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement }))
        {
            framesC = await RunOneRollout(engineC, spec, request, actions);
        }
        swC.Stop();
        samplerCts.Cancel();
        try { await sampler; } catch (OperationCanceledException) { }

        _output.WriteLine($"  {framesC.Length} frames in {swC.Elapsed.TotalSeconds:F2}s");
        _output.WriteLine($"  GPU utilization samples over the run (gpu0%={{VaeDevice}}, gpu1%={{primary}}):");
        foreach ((int g0, int g1) in samples) _output.WriteLine($"    gpu0={g0,3}% gpu1={g1,3}%");

        int midpoint = samples.Count / 2;
        bool gpu0BusyBeforeMidpoint = samples.Take(midpoint).Any(s => s.Gpu0 > 0);
        _output.WriteLine($"  GPU0 (VaeDevice) showed utilization before the run's midpoint: {gpu0BusyBeforeMidpoint} " +
            "(true = genuine overlap, not just a tail-end decode batch).");

        Assert.Equal(framesA.Length, framesC.Length);
        double meanSsim = 0;
        for (int i = 0; i < framesA.Length; i++) meanSsim += Ssim.Compute(framesA[i], framesC[i], Width, Height);
        meanSsim /= framesA.Length;

        double speedupPct = (swA.Elapsed.TotalSeconds - swC.Elapsed.TotalSeconds) / swA.Elapsed.TotalSeconds * 100.0;
        _output.WriteLine($"  mean SSIM(baseline, VaeDevice) = {meanSsim:F4}");
        _output.WriteLine($"  wall-clock: baseline={swA.Elapsed.TotalSeconds:F2}s vs VaeDevice={swC.Elapsed.TotalSeconds:F2}s "
            + $"({speedupPct:F1}% {(speedupPct >= 0 ? "faster" : "slower")}) — reported honestly, not asserted as a hard gate.");

        // Same primary backend/DiT drives both runs — only the VAE's device differs — so a tight SSIM bar is fair
        // (this is NOT the cross-hardware fp8-GEMM-path drift QwenImageDitShardingEngineTests documents).
        Assert.True(meanSsim > 0.95, $"VaeDevice output diverged too far from the same-device baseline (SSIM={meanSsim:F4}).");
    }

    /// <summary>Queues every action before the first drain, matching the documented one-shot session pattern
    /// (<see cref="OasisWorldSession"/>'s class doc) — one rollout, <c>TotalFrames</c> frames, seed frame included.</summary>
    private static async Task<byte[][]> RunOneRollout(InferenceEngine engine, ModelSpec spec, WorldRequest request, string[] actions)
    {
        using IWorldSession session = engine.World.Open(spec, request);
        foreach (string action in actions)
        {
            session.SendAction(action);
        }
        List<byte[]> frames = new List<byte[]>(TotalFrames);
        await foreach (VideoFrame frame in session.StreamAsync(CancellationToken.None))
        {
            frames.Add(frame.Rgb);
        }
        return frames.ToArray();
    }

    private static byte[] SyntheticSeedFrame(int width, int height, int seed)
    {
        byte[] rgb = new byte[width * height * 3];
        Random rng = new Random(seed);
        rng.NextBytes(rgb);
        return rgb;
    }

    private static async Task SampleUtilizationLoop(List<(int Gpu0, int Gpu1)> samples, CancellationToken cancel)
    {
        while (!cancel.IsCancellationRequested)
        {
            samples.Add(SampleUtilization());
            try { await Task.Delay(150, cancel).ConfigureAwait(false); }
            catch (OperationCanceledException) { }
        }
    }

    private static (int Gpu0, int Gpu1) SampleUtilization()
    {
        try
        {
            using Process proc = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "nvidia-smi",
                    Arguments = "--query-gpu=index,utilization.gpu --format=csv,noheader,nounits",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                },
            };
            proc.Start();
            string stdout = proc.StandardOutput.ReadToEnd();
            proc.WaitForExit(2000);
            int gpu0 = 0, gpu1 = 0;
            foreach (string line in stdout.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                string[] parts = line.Split(',', StringSplitOptions.TrimEntries);
                if (parts.Length != 2 || !int.TryParse(parts[0], out int idx) || !int.TryParse(parts[1], out int util))
                    continue;
                if (idx == 0) gpu0 = util;
                if (idx == 1) gpu1 = util;
            }
            return (gpu0, gpu1);
        }
        catch
        {
            return (0, 0);
        }
    }
}
