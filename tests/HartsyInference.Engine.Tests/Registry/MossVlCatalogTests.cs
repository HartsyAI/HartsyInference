using System.Text.RegularExpressions;
using HartsyInference.Engine.Registry;

namespace HartsyInference.Engine.Tests.Registry;

public sealed class MossVlCatalogTests
{
    private static readonly Regex Sha256 = new("^[0-9a-f]{64}$", RegexOptions.Compiled);
    private static readonly Regex Commit = new("^[0-9a-f]{40}$", RegexOptions.Compiled);

    [Fact]
    public void MossVl_IsACliDrivableTextEntryWhoseFirstAssetIsTheGguf()
    {
        CatalogEntry? entry = ModelCatalog.Find("moss-vl-realtime");

        Assert.NotNull(entry);
        Assert.Equal(Modality.Text, entry.Modality);
        Assert.True(entry.CliDrivable);
        Assert.Equal("transformer", entry.Assets[0].Role);
        Assert.EndsWith(".gguf", entry.Assets[0].RepoPath);
    }

    [Fact]
    public void MossVl_ShipsTheTokenizerAndConfigsBesideTheGgufWithPinnedHashes()
    {
        CatalogEntry entry = ModelCatalog.Find("moss-vl-realtime")!;

        // The GGUF embeds no tokenizer, so the tokenizer and the processors' JSON must land in the same folder.
        foreach (string file in new[] { "tokenizer.json", "config.json", "preprocessor_config.json", "video_preprocessor_config.json" })
            Assert.Contains(entry.Assets, asset => asset.RepoPath == file && asset.TargetSubdir == entry.Assets[0].TargetSubdir);
        Assert.All(entry.Assets, asset => Assert.Matches(Sha256, asset.Sha256!));
        Assert.Equal(entry.Assets.Count, entry.Assets.Select(a => Path.Combine(a.TargetSubdir, a.FileName)).Distinct(StringComparer.OrdinalIgnoreCase).Count());
    }

    [Fact]
    public void MossVl_VariantsArePinnedToCommits()
    {
        CatalogEntry entry = ModelCatalog.Find("moss-vl-realtime")!;

        Assert.Equal(["gguf-q4k", "bf16", "nf4", "fp8"], entry.Variants.Select(v => v.Id).ToArray());
        Assert.All(entry.Variants, v => Assert.Matches(Commit, v.Source.Revision));
        Assert.All(entry.Variants, v => Assert.True(v.Source.TotalBytes > 0 && v.Source.FileCount > 0));
        Assert.All(entry.Variants, v => Assert.True(v.Components.HasFlag(CatalogComponents.Vision)));
    }
}
