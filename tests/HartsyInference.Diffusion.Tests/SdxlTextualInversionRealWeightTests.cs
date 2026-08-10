using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression test for Tier 1.8: <see cref="SdxlRecipePipeline"/> now resolves
/// <c>\0swarmembed:NAME\0end</c> markers (SwarmUI core's own rewrite of <c>&lt;embed:name&gt;</c>, arriving on
/// <c>ImageRequest.Prompt</c>/<c>NegativePrompt</c> before this recipe ever sees them — no extension-side change
/// needed) via <c>EmbeddingResolver</c> and threads the resolved inline-embedding map into
/// <c>ClipTextEncoder.EncodePenultimate</c> through <c>EmbeddingResolver.BuildDualClipSchedule</c>. Both
/// <c>EmbeddingResolver</c> and the <c>ClipTextEncoder</c> inline-embedding parameter already existed with zero
/// production callers before this — this test proves the wiring, not new encoder machinery. Same-seed A/B (embed
/// marker present vs. absent), covering BOTH the positive prompt and — the embed's actual real-world use, given it
/// is literally an "ENDXL_Neg" negative embedding — the negative prompt, which needed
/// <see cref="EmbeddingResolver.BuildDualClipSchedule"/>'s <c>negativePlan</c> parameter and a disjoint
/// placeholder-id range (caught in review before this test was written: the first cut only resolved
/// <c>request.Prompt</c>, so a negative-prompt embed — the primary use case for this exact fixture — silently
/// did nothing). Real SDXL weights plus a real dual clip_l/clip_g textual-inversion embedding (civitai "SDXL
/// Negative Embedding by E&amp;D", 74 vectors, .pt/pickle format via <c>TextualInversion.Load</c>'s A1111-layout
/// path) — confirms the learned vectors actually reach the transformer's cross-attention, not just that
/// resolution stopped throwing.</summary>
[Trait("Category", "Integration")]
public sealed class SdxlTextualInversionRealWeightTests
{
    private const string EmbedPath = "/home/hartsy/Desktop/HartsyInference/Models/Embeddings/endxl_neg.pt";
    private readonly ITestOutputHelper _output;
    public SdxlTextualInversionRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void SdxlRecipe_WithPositiveEmbedMarker_ProducesVisiblyDifferentOutputThanWithout()
    {
        if (!Ready(out string ptxDir)) return;

        ImageRequest baseRequest = new ImageRequest
        {
            Prompt = "a photo of a mountain landscape at sunset",
            Width = 512,
            Height = 512,
            Steps = 8,
            CfgScale = 6.0f,
            Seed = 777,
        };
        ImageRequest withEmbedRequest = baseRequest with
        {
            // The literal marker format EmbeddingResolver's regex matches — SwarmUI core rewrites
            // <embed:name> into this before the request ever reaches the recipe (see class doc).
            Prompt = baseRequest.Prompt + " \0swarmembed:endxl_neg\0end",
        };
        AssertVisiblyDifferent(ptxDir, baseRequest, withEmbedRequest, "positive-prompt");
    }

    [Fact]
    public void SdxlRecipe_WithNegativeEmbedMarker_ProducesVisiblyDifferentOutputThanWithout()
    {
        if (!Ready(out string ptxDir)) return;

        ImageRequest baseRequest = new ImageRequest
        {
            Prompt = "a photo of a mountain landscape at sunset",
            NegativePrompt = "blurry, low quality",
            Width = 512,
            Height = 512,
            Steps = 8,
            CfgScale = 6.0f,
            Seed = 777,
        };
        ImageRequest withEmbedRequest = baseRequest with
        {
            NegativePrompt = baseRequest.NegativePrompt + " \0swarmembed:endxl_neg\0end",
        };
        AssertVisiblyDifferent(ptxDir, baseRequest, withEmbedRequest, "negative-prompt");
    }

    private bool Ready(out string ptxDir)
    {
        ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!File.Exists(TestPaths.Sdxl.SingleFile))
        {
            _output.WriteLine($"SKIPPED: SDXL checkpoint not found at {TestPaths.Sdxl.SingleFile}.");
            return false;
        }
        if (!File.Exists(EmbedPath))
        {
            _output.WriteLine($"SKIPPED: textual-inversion embed not found at {EmbedPath}.");
            return false;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA not available.");
            return false;
        }
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found at {ptxDir}.");
            return false;
        }
        return true;
    }

    private void AssertVisiblyDifferent(string ptxDir, ImageRequest without, ImageRequest with, string label)
    {
        byte[] withoutEmbed = Generate(ptxDir, without);
        _output.WriteLine($"[{label}] Generated without embed marker ({withoutEmbed.Length} bytes).");
        byte[] withEmbed = Generate(ptxDir, with);
        _output.WriteLine($"[{label}] Generated with embed marker ({withEmbed.Length} bytes).");
        string offPath = Path.Combine(RepoRoot.Path, $"sdxl_embed_{label}_off.rgb");
        string onPath = Path.Combine(RepoRoot.Path, $"sdxl_embed_{label}_on.rgb");
        File.WriteAllBytes(offPath, withoutEmbed);
        File.WriteAllBytes(onPath, withEmbed);
        _output.WriteLine($"[{label}] Wrote {offPath} and {onPath} for visual inspection.");

        Assert.Equal(withoutEmbed.Length, withEmbed.Length);
        long diffSum = 0;
        for (int i = 0; i < withoutEmbed.Length; i++)
        {
            diffSum += Math.Abs(withoutEmbed[i] - withEmbed[i]);
        }
        double meanAbsDiff = diffSum / (double)withoutEmbed.Length;
        _output.WriteLine($"[{label}] Mean absolute per-byte difference: {meanAbsDiff:F2} (0 would mean byte-identical, i.e. the embed had no effect).");
        Assert.True(meanAbsDiff > 1.0, $"[{label}] Embed-marker vs. no-marker outputs are nearly identical (mean abs diff {meanAbsDiff:F2}) — the resolved embedding likely isn't reaching the transformer.");
    }

    private static byte[] Generate(string ptxDir, ImageRequest request)
    {
        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        RecipeContext context = new RecipeContext
        {
            CheckpointPath = TestPaths.Sdxl.SingleFile,
            Backend = backend,
        };
        using IRecipePipeline pipeline = new SdxlRecipe().Construct(context);
        ImageResult result = pipeline.Generate(request, progress: null, cancel: default);
        return result.Rgb;
    }
}
