using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>End-to-end gate for the distilled default contract. Both failure modes it guards are silent: a
/// routing miss runs the dev contract on a distilled checkpoint, and a broken default falls back to single-pass
/// — either way video still comes out, just the wrong one. Drives the ENGINE with the dev family id over a
/// staged distilled directory, so the filename remap, the default flip, and the two-stage flow are all on the
/// asserted path.</summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "Slow")]
public sealed class LtxVideo25DistilledTwoStageE2ETests
{
    private readonly ITestOutputHelper _output;
    public LtxVideo25DistilledTwoStageE2ETests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task TwoStageRunsByDefaultAndProducesFiniteFrames()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.LtxVideo2.Distilled25,
                TestPaths.LtxVideo2.Gemma4Int8, TestPaths.LtxVideo2.VideoVae25Conv,
                TestPaths.LtxVideo2.AudioVae25, TestPaths.LtxVideo2.LatentUpsampler25)) return;

        // Stage the split checkpoint the way real distilled runs do: transformer-only distilled file + sibling
        // VAEs + Gemma in one directory. The directory NAME is neutral on purpose — the remap must fire off the
        // contained safetensors names, which is the SwarmUI-shaped case.
        string dir = Directory.CreateTempSubdirectory("ltx25-e2e-").FullName;
        try
        {
            File.CreateSymbolicLink(Path.Combine(dir, Path.GetFileName(TestPaths.LtxVideo2.Distilled25)), TestPaths.LtxVideo2.Distilled25);
            File.CreateSymbolicLink(Path.Combine(dir, Path.GetFileName(TestPaths.LtxVideo2.Gemma4Int8)), TestPaths.LtxVideo2.Gemma4Int8);
            File.CreateSymbolicLink(Path.Combine(dir, Path.GetFileName(TestPaths.LtxVideo2.VideoVae25Conv)), TestPaths.LtxVideo2.VideoVae25Conv);
            File.CreateSymbolicLink(Path.Combine(dir, Path.GetFileName(TestPaths.LtxVideo2.AudioVae25)), TestPaths.LtxVideo2.AudioVae25);

            ModelSpec spec = ModelResolver.Resolve("ltx-2.5", dir, Modality.Video);

            // Steps/CfgScale deliberately unset: the DEFAULTS are the thing under test. Geometry kept small and
            // cleanly halvable (320/2/32 = 5 cells, 192/2/32 = 3) so the run stays minutes-cheap and un-snapped.
            VideoRequest request = new VideoRequest
            {
                Prompt = "a red fox trotting through fresh snow at sunrise",
                Width = 320,
                Height = 192,
                Frames = 9,
                Fps = 24,
                Seed = 7,
            };

            int maxStep = 0, totalSteps = 0;
            Progress<StepPreview> progress = new Progress<StepPreview>(p =>
            {
                if (p.Step > maxStep) maxStep = p.Step;
                if (p.TotalSteps > 0) totalSteps = p.TotalSteps;
            });

            using InferenceEngine engine = new InferenceEngine("cuda", 0);
            VideoGenerationResult result = await engine.Video.GenerateAsync(spec, request, progress);

            // 8 base + 3 refine. TotalSteps == 8 here means the default fell back to single-pass; 20 means the
            // remap missed and the dev contract ran.
            Assert.Equal(11, totalSteps);
            Assert.True(maxStep > 8, $"no refine-stage progress fired (max step {maxStep}) — stage 2 did not run");

            Assert.Equal(9, result.Frames.Count);
            Assert.Equal(320, result.Frames[0].Width);
            Assert.Equal(192, result.Frames[0].Height);

            // Finite, non-degenerate output: not black/white, has spatial variance, frames not identical.
            foreach (VideoFrame f in new[] { result.Frames[0], result.Frames[^1] })
            {
                double mean = 0, sq = 0;
                foreach (byte b in f.Rgb) { mean += b; sq += (double)b * b; }
                mean /= f.Rgb.Length;
                double std = Math.Sqrt(Math.Max(0, sq / f.Rgb.Length - mean * mean));
                Assert.InRange(mean, 10, 245);
                Assert.True(std > 5, $"frame std {std:F2} — flat output");
            }
            Assert.False(result.Frames[0].Rgb.AsSpan().SequenceEqual(result.Frames[^1].Rgb),
                "first and last frames are byte-identical — no motion");
            _output.WriteLine($"two-stage e2e ok: {totalSteps} steps (max fired {maxStep}), " +
                $"{result.Frames.Count} frames at {result.Frames[0].Width}x{result.Frames[0].Height}");
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
