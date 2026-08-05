using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>CFG-parallel fallback OBSERVABILITY: the fallback is deliberately silent at the API level (the
/// generation still succeeds), so operators and tests need the <c>[CfgParallel]</c> diagnostic to know which
/// path ran. On this box the enterprise-relevant outcome for Flux is a fallback: the engine's Flux path does
/// not map true-CFG yet (guidance-embedded Dev), so a configured <c>CfgParallelDevice</c> must record
/// <c>inapplicable(no-true-cfg)</c> — and the concurrent path's claim stays explicitly deferred to hardware
/// (2×≥16 GB) plus true-CFG request mapping. What this asserts: the generation succeeds AND the decision line
/// is emitted, i.e. a misconfigured second GPU can never silently pretend to help.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class FluxCfgParallelFallbackTests
{
    private readonly ITestOutputHelper _output;
    public FluxCfgParallelFallbackTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task CfgParallelConfigured_NoTrueCfgRequest_RecordsInapplicableAndStillGenerates()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.Flux.Dev;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;

        ModelSpec spec = ModelResolver.Resolve("flux1", checkpoint, Modality.Image);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: flux1 not resolvable with the explicit path."); return; }

        List<string> captured = new();
        Logs.SetLogger((level, message) =>
        {
            lock (captured) captured.Add(message);
            Console.Error.WriteLine($"[{level}] {message}");
        });
        try
        {
            ImageRequest request = new ImageRequest
            {
                Prompt = "a photograph of an astronaut riding a horse",
                Width = 512,
                Height = 512,
                Steps = 4,
                CfgScale = 1.0f,
                Seed = 42,
            };
            PlacementConfig placement = new PlacementConfig { CfgParallelDevice = "cuda:1" };
            ImageResult result;
            using (InferenceEngine engine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement }))
            {
                result = await engine.Images.GenerateAsync(spec, request);
            }

            int nonZero = result.Rgb.Count(b => b != 0);
            Assert.True(nonZero > result.Rgb.Length * 0.1, "fallback generation produced an all-black image");

            string[] decisions;
            lock (captured) decisions = captured.Where(m => m.Contains("[CfgParallel]")).ToArray();
            _output.WriteLine($"CfgParallel decisions observed: {string.Join(" | ", decisions)}");
            Assert.True(decisions.Length > 0,
                "no [CfgParallel] decision was logged — a configured second GPU silently did nothing, which is "
                + "exactly the observability gap this diagnostic exists to close.");
            Assert.Contains(decisions, d => d.Contains("inapplicable(no-true-cfg)") || d.Contains("fell-back"));
        }
        finally
        {
            Logs.SetLogger(static (level, message) => Console.Error.WriteLine($"[{level}] {message}"));
        }
    }
}
