using HartsyInference.Engine.Registry;

namespace HartsyInference.Engine.Tests.Registry;

public sealed class ModelDownloaderCheckpointFolderTests : IDisposable
{
    private readonly string _folder = Path.Combine(Path.GetTempPath(), "hartsy-ckpt-folder-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_folder))
            Directory.Delete(_folder, recursive: true);
    }

    private static CatalogEntry Sam3d() => ModelCatalog.Find("sam3d-objects") ?? throw new InvalidOperationException("The SAM 3D catalog entry is missing.");

    private void Place(params ModelAsset[] assets)
    {
        Directory.CreateDirectory(_folder);
        foreach (ModelAsset asset in assets)
            File.WriteAllBytes(Path.Combine(_folder, asset.FileName), [0]);
    }

    [Fact]
    public void Sam3dEntry_LoadsFromFolder()
    {
        Assert.True(Sam3d().LoadsFromFolder);
    }

    [Fact]
    public void MissingAssets_AFolderHoldingEveryFileIsTheInstallWhereverTheCatalogWouldPutIt()
    {
        CatalogEntry entry = Sam3d();
        Place([.. entry.Assets]);

        Assert.Empty(ModelDownloader.MissingAssets(entry, _folder));
    }

    [Fact]
    public void MissingAssets_ListsExactlyTheFilesTheFolderLacks()
    {
        CatalogEntry entry = Sam3d();
        IReadOnlyList<ModelAsset> baseline = ModelDownloader.MissingAssets(entry);
        ModelAsset[] present = [.. entry.Assets.Take(2)];
        Place(present);

        IReadOnlyList<ModelAsset> missing = ModelDownloader.MissingAssets(entry, _folder);

        Assert.Equal(baseline.Except(present), missing);
    }

    [Fact]
    public void MissingAssets_NoFolderOrAFolderThatDoesNotExistChangesNothing()
    {
        CatalogEntry entry = Sam3d();
        IReadOnlyList<ModelAsset> baseline = ModelDownloader.MissingAssets(entry);

        Assert.Equal(baseline, ModelDownloader.MissingAssets(entry, null));
        Assert.Equal(baseline, ModelDownloader.MissingAssets(entry, _folder));
    }

    [Fact]
    public void MissingAssets_AnEntryWhoseLoaderReadsTheCatalogFoldersIgnoresTheFolder()
    {
        CatalogEntry entry = ModelCatalog.All.First(e => !e.LoadsFromFolder && e.Assets.Count > 0);
        IReadOnlyList<ModelAsset> baseline = ModelDownloader.MissingAssets(entry);
        Place([.. entry.Assets]);

        Assert.Equal(baseline, ModelDownloader.MissingAssets(entry, _folder));
    }
}
