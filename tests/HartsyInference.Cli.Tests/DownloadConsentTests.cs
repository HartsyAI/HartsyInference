using HartsyInference.Cli.Infra;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>Who may say yes to a download: a person at a terminal, or the setting. A closed or redirected input is a no, and a no holds for the whole run.</summary>
public sealed class DownloadConsentTests
{
    [Fact]
    public void WithoutATerminal_TheAnswerIsNo_AndNothingIsAsked()
    {
        Assert.False(ModelAcquisition.MayDownload(canAsk: false, static () => throw new InvalidOperationException("asked")));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AtATerminal_TheUserDecides(bool answer)
    {
        Assert.Equal(answer, ModelAcquisition.MayDownload(canAsk: true, () => answer));
    }

    [Fact]
    public void TheSetting_AllowsARunWithoutATerminal_AndRefusesAtOne()
    {
        using (KnobProfileScope.Push(KnobProfile.Create("yes").With(EngineKnobs.AllowDownloads, true)))
            Assert.True(ModelAcquisition.MayDownload(canAsk: false, static () => throw new InvalidOperationException("asked")));
        using (KnobProfileScope.Push(KnobProfile.Create("no").With(EngineKnobs.AllowDownloads, false)))
            Assert.False(ModelAcquisition.MayDownload(canAsk: true, static () => throw new InvalidOperationException("asked")));
    }

    [Fact]
    public async Task ADeclinedModel_RefusesTheLoadsOwnDownloads_UntilTheRunEnds()
    {
        // The bug: the spec came back unchanged, the run went on, and the engine fetched the files it had been told not to.
        ModelAsset asset = new()
        {
            Repo = "hartsy-tests/none", RepoPath = "weights.safetensors", TargetSubdir = $"hartsy-tests-{Guid.NewGuid():N}", Role = "transformer",
        };
        ModelSpec spec = new()
        {
            Requested = "consent-test", Modality = Modality.Image,
            Catalog = new CatalogEntry { Id = "consent-test", Modality = Modality.Image, DisplayName = "Consent test", Architecture = "none", Status = ModelStatus.ValidationPending, Assets = [asset] },
        };

        using (KnobProfileScope.Push(KnobProfile.Create("no").With(EngineKnobs.AllowDownloads, false)))
        using (AcquiredModel acquired = ModelAcquisition.EnsurePresent(spec))
        {
            Assert.Null(acquired.Spec.LocalPath);
            Task<Exception?> load;
            using (ExecutionContext.SuppressFlow())
                load = Task.Run(static Exception? () => Record.Exception(static () => DownloadPermission.Demand("a file the load wants")));
            Assert.Contains("paths.allowDownloads=true", Assert.IsType<DownloadRefusedException>(await load).Message);
        }
        DownloadPermission.Demand("after the run");
    }
}
