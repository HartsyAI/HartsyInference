using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>Pins each catalog asset's resolved local path (<c>TargetSubdir</c> + <c>FileName</c>, exactly what
/// <see cref="ModelDownloader"/> combines to decide whether a model is already on disk) against the real path a
/// human verified the file lives at. Same bug class as the Orpheus/SNAC fix (<c>AudioAssetRepoPathTests</c>):
/// `krea2`'s `TargetSubdir` was missing a `/Turbo` segment, and `zimage`'s pointed at `Stable-Diffusion/ZImage/`
/// under the HF repo's own filename, when the file actually present locally is flat under `Stable-Diffusion/`
/// under a different name entirely -- so `ModelDownloader.MissingAssets` always reported both "needs 1 file(s)
/// not on disk" even though the file was right there, and `hartsy image -m krea2`/`-m zimage` only ever worked
/// via an explicit `--model-path` that bypassed the catalog check.</summary>
/// <remarks>This can't be checked by re-deriving the truth mechanically: there's no local HF-repo-layout probe
/// for image checkpoints the way `AudioCheckpoints` has for audio (and even that only resolves a repo's layout
/// by hitting the network, which a unit test must not do). So this pins the path a human verified against the
/// real repo (HF API tree listing) and the real local disk, the same way <see cref="ModelAsset.Sha256"/> pins a
/// file's content. A future hand-edit of either row without re-verifying both sides is exactly the class of
/// mistake this test exists to catch.</remarks>
public sealed class ImageCatalogTargetPathTests
{
    [Fact]
    public void Krea2TransformerResolvesUnderTheTurboSubfolder()
    {
        ModelAsset asset = Transformer("krea2");

        Assert.Equal("Comfy-Org/Krea-2", asset.Repo);
        Assert.Equal("diffusion_models/krea2_turbo_fp8_scaled.safetensors", asset.RepoPath);
        // TargetSubdir is a single forward-slash-joined string (every catalog entry's is), so comparing it
        // against a Path.Combine(...)-built string would pass on Linux/Mac and fail on Windows, where
        // Path.Combine uses '\' -- compare the exact segment values instead of a combined path.
        Assert.Equal("Stable-Diffusion/Krea2/Turbo", asset.TargetSubdir);
        Assert.Equal("krea2_turbo_fp8_scaled.safetensors", asset.FileName);
    }

    [Fact]
    public void ZImageTransformerResolvesToItsLocalRenamedFile()
    {
        ModelAsset asset = Transformer("zimage");

        Assert.Equal("mcmonkey/swarm-models", asset.Repo);
        Assert.Equal("SwarmUI_Z-Image-Turbo-FP8Mix.safetensors", asset.RepoPath);
        // TargetName deliberately differs from the HF repo's own filename -- see the class doc. Compared as
        // exact segment values, not a combined path -- see the comment in the krea2 case above.
        Assert.Equal("Stable-Diffusion", asset.TargetSubdir);
        Assert.Equal("z-image-turbo.safetensors", asset.TargetName);
        Assert.Equal("z-image-turbo.safetensors", asset.FileName);
    }

    private static ModelAsset Transformer(string catalogId)
    {
        CatalogEntry? entry = ModelCatalog.Find(catalogId);
        Assert.NotNull(entry);
        return Assert.Single(entry!.Assets, a => a.Role == "transformer");
    }
}
