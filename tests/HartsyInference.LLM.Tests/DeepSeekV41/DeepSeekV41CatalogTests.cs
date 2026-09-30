using HartsyInference.Core.Exceptions;
using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.ModelAssets.Quant;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The <c>deepseek-v4.1-flash</c> catalog row: pinned revisions, per-derivative components and the MLX draft refusal.</summary>
public sealed class DeepSeekV41CatalogTests : IDisposable
{
    private readonly string _directory = Directory.CreateTempSubdirectory("dsv41-catalog-").FullName;

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    private static CatalogVariant Variant(string id) =>
        ModelCatalog.Find(DeepSeekV41Catalog.Id)!.Variants.Single(variant => variant.Id == id);

    [Fact]
    public void TheRowIsInTheCatalogButOffersNoDownload()
    {
        CatalogEntry? entry = ModelCatalog.Find(DeepSeekV41Catalog.Id);

        Assert.NotNull(entry);
        Assert.Equal(Modality.Text, entry!.Modality);
        Assert.False(entry.CliDrivable);
        Assert.True(entry.Assets is null or { Count: 0 });
        Assert.Equal(7, entry.Variants.Count);
    }

    [Fact]
    public void OfficialVariant_IsPinnedToTheInspectedRevision()
    {
        CatalogShardSet source = Variant("official").Source;

        Assert.Equal("deepseek-ai/DeepSeek-V4.1-Flash", source.Repo);
        Assert.Equal("dba1be0a40aa45a94ad051997016db3960a90277", source.Revision);
        Assert.Equal(48, source.FileCount);
        Assert.Equal(QuantFlavor.Official, Variant("official").Flavor);
    }

    [Fact]
    public void EveryVariantPinsAFullCommitShaAndAFileCount()
    {
        foreach (CatalogVariant variant in ModelCatalog.Find(DeepSeekV41Catalog.Id)!.Variants)
        {
            Assert.Matches("^[0-9a-f]{40}$", variant.Source.Revision);
            Assert.True(variant.Source.FileCount > 0, variant.Id);
            Assert.True(variant.Source.TotalBytes > 0, variant.Id);
            Assert.True(variant.Components.HasFlag(CatalogComponents.Backbone), variant.Id);
        }
    }

    [Fact]
    public void ComponentFlagsDifferPerDerivative()
    {
        Assert.True(Variant("official").Components.HasFlag(CatalogComponents.Draft));
        Assert.True(Variant("quark-mxfp4").Components.HasFlag(CatalogComponents.Vision));
        Assert.False(Variant("mlx-4bit").Components.HasFlag(CatalogComponents.Draft));
        Assert.True(Variant("mlx-4bit").Components.HasFlag(CatalogComponents.Engram));
    }

    [Theory]
    [InlineData("dwarfstar-q2")]
    [InlineData("dwarfstar-q4")]
    public void DwarfStarHasNoDraftAndShipsVisionAsASeparateFile(string id)
    {
        CatalogVariant variant = Variant(id);

        Assert.False(variant.Components.HasFlag(CatalogComponents.Draft));
        Assert.True(variant.VisionSeparateFile);
        Assert.Contains("no draft", variant.Refusals[CatalogComponents.Draft]);
    }

    [Fact]
    public void MlxRefusesTheDraftWithTheExactMissingExpertList()
    {
        Assert.Equal(DeepSeekV41Catalog.MlxDraftRefusal, Variant("mlx-4bit").Refusals[CatalogComponents.Draft]);
        Assert.Equal("DSpark speculative decoding refused: mtp.2 lacks 14 of 128 routed experts (9, 87-99).",
            DeepSeekV41Catalog.MlxDraftRefusal);
    }

    [Fact]
    public void MlxCatalogRefusalMatchesWhatTheCheckpointScanReportsForTheSameGapShape()
    {
        // The real repo lacks experts 9, 87 (weights only) and 88-99 of 128 in mtp.2; the scan must render that identically.
        Dictionary<int, IReadOnlyList<int>> missing = new() { [2] = new[] { 9, 87 }.Concat(Enumerable.Range(88, 12)).ToArray() };
        DeepSeekV41DraftReport report = new(DeepSeekV41DraftStatus.Incomplete, missing, 128);

        Assert.Equal(DeepSeekV41Catalog.MlxDraftRefusal, $"DSpark speculative decoding refused: {report.DescribeMissing()}.");
    }

    [Fact]
    public void MlxRefusalSurfacesFromARealCheckpointOpen()
    {
        TinyDeepSeekV41Checkpoint.Write(_directory, QuantFlavor.Mlx, fullDraftExperts: [0], partialDraftExpert: 1);

        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory);
        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(checkpoint.RequireDraft);

        Assert.Contains("DSpark speculative decoding is refused", error.Message);
        Assert.Contains(checkpoint.Draft.DescribeMissing(), error.Message);
    }
}
