using HartsyInference.Cli.Infra;
using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>Pins each audio-cache asset's <see cref="ModelAsset.RepoPath"/> against the real filename its
/// HuggingFace repo ships under, so a hand-edited catalog entry can't silently drift from reality the way the
/// Orpheus/SNAC entry did: <c>ModelCatalog</c> declared <c>hubertsiuzdak/snac_24khz</c>'s codec as
/// <c>model.safetensors</c>, which that repo does not have (it ships <c>pytorch_model.bin</c> only) — so
/// <see cref="ModelAcquisition.EnsureAudioAssetsPresent"/> reported the file "missing" on every single call and
/// re-prompted "Download these now?" forever, even after a successful download.</summary>
/// <remarks>This can't be checked by re-deriving the truth mechanically in a unit test: the real loaders
/// (<c>AudioCheckpoints.ResolveCheckpointFilesAsync</c>) discover a repo's actual layout by PROBING it over the
/// network at load time (single-file safetensors, else a shard index, else a pickle) — exactly the thing a unit
/// test must not do. So this pins the filename a human has actually verified against the live repo, the same way
/// <see cref="ModelAsset.Sha256"/> pins a file's content. A future hand-edit of one of these rows without
/// re-verifying against the real repo is exactly the class of mistake this test exists to catch.
///
/// <para>Scope: the two assets actually involved in the bug this test guards (Orpheus's backbone + SNAC codec),
/// not every audio-cache catalog entry — extending the <see cref="VerifiedRepoPaths"/> table to more models is
/// straightforward once each one's real layout is verified the same way, but asserting an unverified guess would
/// make this test actively misleading.</para></remarks>
public sealed class AudioAssetRepoPathTests
{
    /// <summary>(catalog id, asset role) → the <see cref="ModelAsset.RepoPath"/> verified against that repo's
    /// real HuggingFace file listing.</summary>
    private static readonly Dictionary<(string CatalogId, string Role), string> VerifiedRepoPaths = new()
    {
        // unsloth/orpheus-3b-0.1-ft ships a single-file model.safetensors at its repo root.
        [("orpheus", "transformer")] = "model.safetensors",
        // hubertsiuzdak/snac_24khz has neither a model.safetensors nor a shard index -- it ships pytorch_model.bin
        // only. This is the exact row that was wrong (as "model.safetensors") before the Orpheus/SNAC fix.
        [("orpheus", "codec")] = "pytorch_model.bin",
    };

    public static IEnumerable<object[]> VerifiedCases() =>
        VerifiedRepoPaths.Select(kv => new object[] { kv.Key.CatalogId, kv.Key.Role, kv.Value });

    [Theory]
    [MemberData(nameof(VerifiedCases))]
    public void CatalogRepoPathMatchesTheVerifiedRealFilename(string catalogId, string role, string expectedRepoPath)
    {
        CatalogEntry? entry = ModelCatalog.Find(catalogId);
        Assert.NotNull(entry);

        ModelAsset asset = Assert.Single(entry!.Assets, a => a.Role == role);
        Assert.Equal(expectedRepoPath, asset.RepoPath);
    }

    [Fact]
    public void OrpheusAssetsResolveThroughTheAudioCachePath()
    {
        // Both orpheus assets are TTS-modality and not in the standard-download exemption list, so they must
        // resolve through ModelAcquisition's audio-cache branch (AudioModelCache), not ModelDownloader's
        // Models/<subdir> tree -- otherwise the RepoPath pins above would be checking the wrong code path, and
        // a real mismatch on this entry would surface as a disk-tree problem instead of a "download loop" one.
        CatalogEntry entry = ModelCatalog.Find("orpheus")!;

        Assert.True(ModelAcquisition.UsesAudioCache(entry, entry.Modality));
    }
}
