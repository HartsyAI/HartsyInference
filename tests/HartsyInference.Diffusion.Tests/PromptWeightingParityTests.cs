using Xunit;
using Xunit.Abstractions;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The two end-to-end properties SwarmUI's <c>(word:1.5)</c> parity rests on, at the level where the
/// conditioning cache actually lives — which is the level a unit test cannot reach.
/// <para><b>The cache hazard these exist for:</b> under
/// <see cref="Diffusion.Prompting.PromptWeightingMode.CondScale"/> a weighted prompt tokenizes to the SAME ids as the
/// unweighted one, and the pipelines key their cross-generation conditioning cache on token ids alone. So a weighted
/// generation followed by a plain one is the case where an in-place scale, or a cached weighted tensor, shows up — as a
/// plain prompt silently inheriting the previous request's emphasis. Nothing about the weighted image itself would look
/// wrong.</para></summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class PromptWeightingParityTests
{
    private readonly ITestOutputHelper _output;

    public PromptWeightingParityTests(ITestOutputHelper output) => _output = output;

    private static ImageRequest Request(string prompt) => new ImageRequest
    {
        Prompt = prompt,
        Width = 1024,
        Height = 1024,
        Steps = 8,
        CfgScale = 1.0f,
        Seed = 42,
    };

    /// <summary>Weight 1 must be the plain prompt, byte for byte: the emphasis grammar resolves to all-ones weights,
    /// every mechanism short-circuits, and nothing is multiplied by one through a device round trip.</summary>
    [Fact]
    public async Task AWeightOfOneIsByteIdenticalToThePlainPrompt()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string checkpoint = TestPaths.QwenImage.V1;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;
        ModelSpec spec = ModelResolver.Resolve("qwen-image", checkpoint, Modality.Image);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: qwen-image not resolvable with the explicit path."); return; }

        using InferenceEngine engine = new InferenceEngine("cuda", 0);
        ImageResult plain = await engine.Images.GenerateAsync(spec, Request("a photograph of a red cat"));
        ImageResult weightedOne = await engine.Images.GenerateAsync(spec, Request("a photograph of a (red:1.0) cat"));

        Assert.Equal(plain.Rgb, weightedOne.Rgb);
    }

    /// <summary>The cache invariant: plain, then weighted, then plain again. The third result must equal the first.
    /// A failure here means the weighted conditioning reached the cache the token-id key describes, so every later
    /// generation on that prompt carries an emphasis nobody asked for.</summary>
    [Fact]
    public async Task AWeightedGenerationDoesNotLeakIntoTheNextPlainOne()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string checkpoint = TestPaths.QwenImage.V1;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;
        ModelSpec spec = ModelResolver.Resolve("qwen-image", checkpoint, Modality.Image);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: qwen-image not resolvable with the explicit path."); return; }

        using InferenceEngine engine = new InferenceEngine("cuda", 0);
        ImageResult first = await engine.Images.GenerateAsync(spec, Request("a photograph of a red cat"));
        ImageResult weighted = await engine.Images.GenerateAsync(spec, Request("a photograph of a (red:1.6) cat"));
        ImageResult third = await engine.Images.GenerateAsync(spec, Request("a photograph of a red cat"));

        Assert.Equal(first.Rgb, third.Rgb);
        // And the weighting has to have done something, or the invariant above is satisfied trivially.
        double ssim = Ssim.Compute(first.Rgb, weighted.Rgb, weighted.Width, weighted.Height);
        _output.WriteLine($"SSIM(plain, weighted 1.6) = {ssim:F4} — expected clearly below 1.");
        Assert.True(ssim < 0.999, $"a 1.6 emphasis changed nothing (SSIM={ssim:F4}); the weights are not reaching the "
            + "conditioning — check that the recipe declares a PromptWeightingMode and that the scale runs after any "
            + "template trim.");
    }
}
