using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using HartsyInference.LLM.DeepSeekV41;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The <c>deepseek-v4.1-flash</c> catalog row: pinned revisions, per-derivative components and the MLX draft refusal.</summary>
public sealed class DeepSeekV41CatalogTests
{
    private static CatalogVariant Variant(string id) =>
        ModelCatalog.Find(DeepSeekV41Catalog.Id)!.Variants.Single(variant => variant.Id == id);

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
    public void DwarfStarHasNoDraftAndShipsVisionAsASeparateFile()
    {
        CatalogVariant variant = Variant("dwarfstar-q2");

        Assert.False(variant.Components.HasFlag(CatalogComponents.Draft));
        Assert.True(variant.VisionSeparateFile);
        Assert.Contains("no draft", variant.Refusals[CatalogComponents.Draft]);
    }

    [Fact]
    public void MlxCatalogRefusalMatchesWhatTheCheckpointScanReportsForTheSameGapShape()
    {
        // The real repo lacks experts 9, 87 (weights only) and 88-99 of 128 in mtp.2; the scan must render that identically.
        Dictionary<int, IReadOnlyList<int>> missing = new() { [2] = new[] { 9, 87 }.Concat(Enumerable.Range(88, 12)).ToArray() };
        DeepSeekV41DraftReport report = new(DeepSeekV41DraftStatus.Incomplete, missing, 128);

        Assert.Equal(DeepSeekV41Catalog.MlxDraftRefusal, $"DSpark speculative decoding refused: {report.DescribeMissing()}.");
    }
}
