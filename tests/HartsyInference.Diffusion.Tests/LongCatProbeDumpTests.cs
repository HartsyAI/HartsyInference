using HartsyInference.Core.Configuration;
using HartsyInference.Tests.Shared;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The avatar DiT probe dump (<c>diagnostics.longcatProbeDump</c>): the first denoise step of the first call writes the embedders, the RoPE
/// tables, every block output and, for the named blocks, each stage.</summary>
[Trait("Category", "SyntheticSmoke")]
[Collection(GlobalKnobStore.Name)]
public sealed class LongCatProbeDumpTests
{
    [Fact]
    public void BlockListParsesAndBlankMeansNoStages()
    {
        Assert.Null(LongCatProbeDump.ParseBlocks(null));
        Assert.Null(LongCatProbeDump.ParseBlocks("  "));
        Assert.Equal(new HashSet<int> { 0, 2, 47 }, LongCatProbeDump.ParseBlocks("0, 2,47"));
    }

    [Fact]
    public void TheFirstStepWritesEmbeddersBlockOutputsAndTheNamedBlocksStages()
    {
        string dir = Path.Combine(Path.GetTempPath(), "longcat-probe-" + Guid.NewGuid().ToString("N"));
        using (KnobProfile.Create("longcat-probe-dump")
                   .With(EngineKnobs.LongCatProbeDump, dir)
                   .With(EngineKnobs.LongCatProbeBlocks, "0")
                   .Push())
        {
            LongCatAvatarRecipeEndToEndTests.Run(onePointSix: false, audioSeed: 1);
        }

        string[] names = Directory.GetFiles(dir).Select(Path.GetFileName).Select(n => n!).ToArray();
        Assert.Contains("call00_probe_x_embedder.npy", names);
        Assert.Contains("call00_probe_block0.out.npy", names);
        Assert.Contains("call00_probe_block1.out.npy", names);
        // The named block dumps every stage; the other blocks dump only their output.
        Assert.Contains("call00_probe_block0.modulation.npy", names);
        Assert.DoesNotContain("call00_probe_block1.modulation.npy", names);
        Directory.Delete(dir, recursive: true);
    }
}
