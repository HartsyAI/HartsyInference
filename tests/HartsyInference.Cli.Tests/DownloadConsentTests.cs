using HartsyInference.Cli.Infra;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>Keeps the process-wide refusal of these tests away from the rest of the assembly.</summary>
[CollectionDefinition("DownloadPermission", DisableParallelization = true)]
public sealed class DownloadPermissionCollection;

/// <summary>Who may say yes to a download: a person at a terminal, or the setting. A closed or redirected input is a no, and a no holds for the whole run.</summary>
[Collection("DownloadPermission")]
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
    public void TheSetting_IsTheAnswer_AndNobodyIsAsked()
    {
        // A "no" typed against a setting that says yes would not hold for the load, so the question is not put.
        foreach (bool canAsk in new[] { true, false })
        {
            foreach (bool setting in new[] { true, false })
            {
                using (KnobProfileScope.Push(KnobProfile.Create("set").With(EngineKnobs.AllowDownloads, setting)))
                    Assert.Equal(setting, ModelAcquisition.MayDownload(canAsk, static () => throw new InvalidOperationException("asked")));
            }
        }
    }

    [Theory]
    [InlineData(false, null, "no terminal")]
    [InlineData(true, null, null)]
    [InlineData(false, true, null)]
    [InlineData(true, false, "paths.allowDownloads is off")]
    [InlineData(false, false, "paths.allowDownloads is off")]
    public async Task TheRefusalAtStart_HoldsOnAThreadTheSettingDoesNotReach(bool canAsk, bool? setting, string? refusal)
    {
        KnobProfile profile = KnobProfile.Create("set");
        if (setting is bool value)
            profile = profile.With(EngineKnobs.AllowDownloads, value);
        using (KnobProfileScope.Push(profile))
        using (ModelAcquisition.RefuseAtStart(canAsk))
        {
            Exception? here = Record.Exception(static () => DownloadPermission.Demand("x")), there = await OnAnotherThread("x");
            if (refusal is null)
            {
                Assert.Null(here);
                Assert.Null(there);
            }
            else
            {
                Assert.Contains(refusal, Assert.IsType<DownloadRefusedException>(here).Message);
                Assert.Contains(refusal, Assert.IsType<DownloadRefusedException>(there).Message);
            }
        }
        DownloadPermission.Demand("after the run");
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
            // The profile does not reach this thread; what stops the load there is the refusal EnsurePresent left in force.
            Assert.IsType<DownloadRefusedException>(await OnAnotherThread("a file the load wants"));
        }
        DownloadPermission.Demand("after the run");
    }

    private static Task<Exception?> OnAnotherThread(string what)
    {
        using (ExecutionContext.SuppressFlow())
            return Task.Run(Exception? () => Record.Exception(() => DownloadPermission.Demand(what)));
    }
}
