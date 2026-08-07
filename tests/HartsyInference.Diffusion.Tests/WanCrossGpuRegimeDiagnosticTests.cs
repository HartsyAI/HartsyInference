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

/// <summary>DIAGNOSTIC (not a regression gate): the missing control for interpreting Wan context-parallel
/// SSIM numbers — the WHOLE model, no CP code at all, generated once on ordinal 0 (4090, SM 8.9) and once on
/// ordinal 1 (3060, SM 8.6), identical request/seed. Whatever SSIM this measures is the pure cross-architecture
/// numeric-drift ceiling for this fp16 checkpoint on this box: a CP run that mixes the two cards can never be
/// expected to beat it. Lesson encoded from the H3 mosaic history: this control's outputs are COHERENCE-checked
/// (not just scored) so a broken-on-one-card render can't masquerade as "drift".</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class WanCrossGpuRegimeDiagnosticTests
{
    private readonly ITestOutputHelper _output;
    public WanCrossGpuRegimeDiagnosticTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task WholeModel_Ordinal0_Vs_Ordinal1_MeasuresCrossArchDriftCeiling()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.WanVideo.Ti2V5B, TestPaths.WanVideo.Umt5Xxl)) return;
        foreach (GpuTopologyInfo gpu in CudaTopology.Probe())
        {
            _output.WriteLine($"GPU {gpu.Ordinal} ({gpu.Name}): {gpu.FreeMemoryBytes >> 20} MiB free");
            if (gpu.FreeMemoryBytes < 11L << 30)
            {
                _output.WriteLine($"SKIPPED: GPU {gpu.Ordinal} has < 11 GiB free.");
                return;
            }
        }

        ModelSpec spec = ModelResolver.Resolve("wan", TestPaths.WanVideo.Ti2V5B, Modality.Video);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: wan not resolvable with the explicit path."); return; }

        // Byte-identical to WanContextParallelEngineTests' request so the numbers compose directly.
        VideoRequest request = new VideoRequest
        {
            Prompt = "a red ball bouncing on a wooden table",
            Width = 480,
            Height = 480,
            Frames = 9,
            Steps = 6,
            CfgScale = 5.0f,
            Seed = 42,
        };

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine("[1/2] Whole model on ordinal 0...");
        VideoGenerationResult on0;
        using (InferenceEngine engine = new InferenceEngine("cuda", 0))
        {
            on0 = await engine.Video.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  ordinal 0: {on0.Frames.Count} frames, {sw.Elapsed.TotalSeconds:F1}s");

        sw.Restart();
        _output.WriteLine("[2/2] Whole model on ordinal 1...");
        VideoGenerationResult on1;
        using (InferenceEngine engine = new InferenceEngine("cuda", 1))
        {
            on1 = await engine.Video.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  ordinal 1: {on1.Frames.Count} frames, {sw.Elapsed.TotalSeconds:F1}s");

        Assert.Equal(on0.Frames.Count, on1.Frames.Count);
        foreach ((VideoGenerationResult result, string label) in new[] { (on0, "ordinal0"), (on1, "ordinal1") })
        {
            VideoFrame f = result.Frames[result.Frames.Count / 2];
            int nonZero = f.Rgb.Count(v => v != 0), nonFF = f.Rgb.Count(v => v != 255);
            Assert.True(nonZero > f.Rgb.Length / 10, $"{label}: middle frame is all-black");
            Assert.True(nonFF > f.Rgb.Length / 10, $"{label}: middle frame is all-white");
        }
        VideoFrame m0 = on0.Frames[on0.Frames.Count / 2];
        VideoFrame m1 = on1.Frames[on1.Frames.Count / 2];
        double ssim = Ssim.Compute(m0.Rgb, m1.Rgb, m0.Width, m0.Height);
        _output.WriteLine($"CROSS-ARCH DRIFT CEILING: middle-frame SSIM(ordinal0, ordinal1) = {ssim:F4}");
        // Coherence floor only — the measured value IS the deliverable; both outputs were coherence-checked above.
        Assert.True(ssim > 0.05, $"cross-card outputs are incoherently different (SSIM={ssim:F4}) — that is a real defect, not drift.");
    }
}
