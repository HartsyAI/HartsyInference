using HartsyInference.Core.Configuration;
using HartsyInference.Engine;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>A generation that stops to tell the operator to go and download a VAE by hand is a generation that
/// failed, so the overload every recipe calls now fetches a missing side model. These pin that default and its
/// escape hatch without touching the network: the asset names a repo that does not exist, so autofetch-off throws
/// <see cref="FileNotFoundException"/> before any request and autofetch-on fails as a download instead — two
/// outcomes the exception type tells apart. Points the models root at a hermetic fixture, so nothing here can
/// resolve to a real file on the machine running it.</summary>
[Collection("ModelsRootKnob")]
public sealed class SideModelAutofetchTests : IDisposable
{
    private readonly string _tempModelsRoot = Path.Combine(Path.GetTempPath(), "hartsy-autofetch-tests-" + Path.GetRandomFileName());

    public SideModelAutofetchTests()
    {
        Directory.CreateDirectory(_tempModelsRoot);
        KnobStore.Set(EngineKnobs.ModelsRoot, _tempModelsRoot);
    }

    public void Dispose()
    {
        KnobStore.Clear(EngineKnobs.ModelsRoot);
        KnobStore.Clear(EngineKnobs.SideModelAutofetch);
        if (Directory.Exists(_tempModelsRoot))
            Directory.Delete(_tempModelsRoot, recursive: true);
    }

    private static ModelAsset Missing() => new()
    {
        Repo = "hartsy-tests/does-not-exist",
        RepoPath = "nothing.safetensors",
        TargetSubdir = "VAE/Flux",
        Role = "vae",
    };

    [Fact]
    public void TheDefaultIsToFetch()
    {
        Assert.True(EngineKnobs.SideModelAutofetch.Default);
    }

    [Fact]
    public async Task AutofetchOff_ThrowsBeforeAnyRequest_NamingTheKnobAndTheRepo()
    {
        KnobStore.Set(EngineKnobs.SideModelAutofetch, false);
        ModelAsset asset = Missing();
        FileNotFoundException ex = await Assert.ThrowsAsync<FileNotFoundException>(
            () => ModelDownloader.EnsureSideModelAsync(asset, onProgress: null, CancellationToken.None));
        Assert.Contains("paths.sideModelAutofetch", ex.Message);
        Assert.Contains(asset.Repo, ex.Message);
    }

    [Fact]
    public async Task AutofetchOn_TriesToDownload_RatherThanTellingTheOperatorToDoIt()
    {
        KnobStore.Set(EngineKnobs.SideModelAutofetch, true);
        Exception? ex = await Record.ExceptionAsync(
            () => ModelDownloader.EnsureSideModelAsync(Missing(), onProgress: null, CancellationToken.None));
        Assert.NotNull(ex);
        Assert.IsNotType<FileNotFoundException>(ex);
    }

    [Fact]
    public async Task AnExplicitFalse_StillWins_ForCallersThatMustNotReachTheNetwork()
    {
        KnobStore.Set(EngineKnobs.SideModelAutofetch, true);
        await Assert.ThrowsAsync<FileNotFoundException>(
            () => ModelDownloader.EnsureSideModelAsync(Missing(), downloadIfMissing: false, onProgress: null, CancellationToken.None));
    }
}
