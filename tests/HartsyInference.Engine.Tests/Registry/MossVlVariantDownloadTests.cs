using System.Text.RegularExpressions;
using HartsyInference.Core.Configuration;
using HartsyInference.Engine.Registry;

namespace HartsyInference.Engine.Tests.Registry;

/// <summary>The MOSS-VL folder variants (bf16, nf4, fp8) as the downloader sees them: file lists, pins, hashes and the shard check.
/// Nothing is downloaded; the network path is only reached for a file that is missing, and every test here has them present or stops first.</summary>
[Collection("ModelsRootKnob")]
public sealed class MossVlVariantDownloadTests : IDisposable
{
    private static readonly Regex Sha256 = new("^[0-9a-f]{64}$", RegexOptions.Compiled);
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-variants-" + Guid.NewGuid().ToString("N"));
    private readonly CatalogEntry _entry = ModelCatalog.Find("moss-vl-realtime")!;

    public MossVlVariantDownloadTests()
    {
        Directory.CreateDirectory(_root);
        KnobStore.Set(EngineKnobs.ModelsRoot, _root);
    }

    public void Dispose()
    {
        KnobStore.Clear(EngineKnobs.ModelsRoot);
        Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void TheEntryOffersTheGgufAndEveryOfficialFolderVariant()
    {
        Assert.Equal(["gguf-q4k", "bf16", "nf4", "fp8"], _entry.Variants.Select(v => v.Id).ToArray());
        // A folder with both a GGUF and a shard index is ambiguous to the loader, so each folder variant gets its own folder.
        string[] folders = [.. _entry.Variants.Where(v => v.Source.IndexFile is not null).Select(v => v.Source.TargetSubdir)];
        Assert.Equal(folders.Length, folders.Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.DoesNotContain(_entry.Assets[0].TargetSubdir, folders, StringComparer.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("nf4", 3, 11_841_998_136)]
    [InlineData("fp8", 4, 15_729_921_752)]
    [InlineData("bf16", 5, 22_672_852_616)]
    public void AFolderVariantListsItsIndexShardsAndSupportFilesAtItsPinnedRevision(string id, int shards, long bytes)
    {
        CatalogShardSet set = ModelDownloader.FindVariant(_entry, id).Source;
        Assert.Equal(shards, set.FileCount);
        Assert.Equal(bytes, set.TotalBytes);

        IReadOnlyList<ModelAsset> files = ModelDownloader.VariantAssets(_entry, id);
        Assert.Equal(1 + shards + set.SupportFiles.Count, files.Count);
        Assert.Equal("model.safetensors.index.json", files[0].RepoPath);
        Assert.Equal(shards, files.Count(f => f.RepoPath.EndsWith(".safetensors", StringComparison.Ordinal)));
        Assert.All(files, f => Assert.Equal(set.TargetSubdir, f.TargetSubdir));
        Assert.All(files, f => Assert.Equal(set.Repo, f.Repo));
        foreach (string needed in new[] { "config.json", "tokenizer.json", "preprocessor_config.json", "video_preprocessor_config.json" })
            Assert.Contains(files, f => f.RepoPath == needed);
        // Every file is verified: the weights by the hub's LFS hash, the small JSON by the hash of the pinned revision.
        Assert.All(files, f => Assert.Matches(Sha256, f.Sha256 ?? ""));
        Assert.Equal(files.Count, files.Select(f => f.FileName).Distinct(StringComparer.Ordinal).Count());
    }

    [Fact]
    public void TheNf4AndFp8FoldersCarryTheirOwnConfigsNotTheBf16Ones()
    {
        string ConfigSha(string id) => ModelDownloader.VariantAssets(_entry, id).Single(f => f.RepoPath == "config.json").Sha256!;
        Assert.NotEqual(ConfigSha("bf16"), ConfigSha("nf4"));
        Assert.NotEqual(ConfigSha("nf4"), ConfigSha("fp8"));
    }

    [Fact]
    public void ASingleFileVariantIsTheDefaultAssetSet() =>
        Assert.Same(_entry.Assets, ModelDownloader.VariantAssets(_entry, "gguf-q4k"));

    [Fact]
    public void AnUnknownVariantIsRefusedWithTheOnesThatExist()
    {
        ArgumentException error = Assert.Throws<ArgumentException>(() => ModelDownloader.FindVariant(_entry, "awq"));
        Assert.Contains("nf4", error.Message);
        Assert.Contains("fp8", error.Message);
        Assert.Equal("NF4", ModelDownloader.FindVariant(_entry, "NF4").Id.ToUpperInvariant());   // case-insensitive
    }

    [Fact]
    public void TheFolderIsKnownOnlyOnceItsIndexIsOnDisk()
    {
        Assert.Null(ModelDownloader.VariantFolder(_entry, "nf4"));
        Assert.Null(ModelDownloader.VariantFolder(_entry, "gguf-q4k"));   // a single file has no folder
        ModelAsset index = ModelDownloader.VariantAssets(_entry, "nf4")[0];
        string path = ModelDownloader.TargetPath(index);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "{}");
        // The models folder may be spelled LLM or llm; the engine matches either.
        Assert.Equal(Path.GetDirectoryName(path), ModelDownloader.VariantFolder(_entry, "nf4"), ignoreCase: true);
    }

    [Fact]
    public async Task AnIndexThatNamesAShardOutsideTheSetFailsTheDownloadCheck()
    {
        // Every listed file is already present, so nothing is fetched; the index then names a shard that does not exist.
        foreach (ModelAsset file in ModelDownloader.VariantAssets(_entry, "nf4"))
        {
            string path = ModelDownloader.TargetPath(file);
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, file.RepoPath == "model.safetensors.index.json"
                ? """{"weight_map":{"a":"model-00001-of-00003.safetensors","b":"model-00009-of-00009.safetensors"}}"""
                : "x");
        }
        InvalidDataException error = await Assert.ThrowsAsync<InvalidDataException>(
            () => ModelDownloader.DownloadVariantAsync(_entry, "nf4", onProgress: null, CancellationToken.None));
        Assert.Contains("model-00009-of-00009.safetensors", error.Message);
    }

    [Fact]
    public async Task ACompleteFolderPassesTheCheckWithoutTouchingTheNetwork()
    {
        foreach (ModelAsset file in ModelDownloader.VariantAssets(_entry, "fp8"))
        {
            string path = ModelDownloader.TargetPath(file);
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, file.RepoPath == "model.safetensors.index.json"
                ? """{"weight_map":{"a":"model-00001-of-00004.safetensors","b":"model-00004-of-00004.safetensors"}}"""
                : "x");
        }
        await ModelDownloader.DownloadVariantAsync(_entry, "fp8", onProgress: null, CancellationToken.None);
    }
}

/// <summary>Serializes the tests that point the process-wide models root at a temp folder.</summary>
[CollectionDefinition("ModelsRootKnob", DisableParallelization = true)]
public sealed class ModelsRootKnobCollection;
