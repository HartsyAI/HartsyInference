using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Duo;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Duo;
using Xunit;

namespace HartsyInference.Engine.Tests.Duo;

/// <summary>The Duo recipe on the CPU with a synthetic release: construction from the release folder, the output mapping, and the refusals.
/// The codecs are fakes and nothing here uses released weights.</summary>
public sealed class DuoRecipeTests : IDisposable
{
    private readonly DuoReleaseFixture _release = new();
    private readonly CpuBackend _backend = new();

    public void Dispose()
    {
        _release.Dispose();
        _backend.Dispose();
    }

    [Fact]
    public void EachReleaseResolvesToItsStageWithItsOwnSamplerDefaults()
    {
        foreach (string causal in new[] { "duo-v1-causal", "duo-v1-2-causal" })
        {
            VideoDefaults defaults = Assert.IsType<DuoRecipe>(VideoRecipeRegistry.Resolve(causal)).Defaults;
            Assert.Equal((50, 1f), (defaults.Steps, defaults.FlowShift!.Value));
        }
        VideoDefaults distilled = Assert.IsType<DuoRecipe>(VideoRecipeRegistry.Resolve("duo-v1-distill")).Defaults;
        Assert.Equal((4, 5f), (distilled.Steps, distilled.FlowShift!.Value));
    }

    [Fact]
    public async Task ADistilledReleaseGeneratesFromARunFolderWithReNoise()
    {
        using DuoReleaseFixture release = new(distilled: true);
        DuoRecipe recipe = new(distilled: true);
        using DuoRecipePipeline pipeline = recipe.Build(
            new RecipeContext { CheckpointPath = release.Release, Backend = _backend }, (_, _) => DuoReleaseFixture.Codecs());
        VideoRequest request = Request((DuoRecipePipeline.RunFolderKey, release.RunFolder)) with { Steps = null };
        VideoEffectiveSettings settings = (await Plan(request, recipe, "duo-v1-distill")).EffectiveSettings;
        List<int> steps = [];

        VideoGenerationResult result = pipeline.Generate(request, settings, new Immediate(p => steps.Add(p.Step)), default);

        Assert.Equal((4, 5f), (settings.Steps, settings.FlowShift!.Value));
        Assert.Equal(DuoReleaseFixture.Frames, result.Frames.Count);
        Assert.Equal(Enumerable.Range(1, 8), steps);
        // Other draws give another clip, so the folder's re-noise reaches the output.
        File.Copy(Path.Combine(release.RunFolder, "rvideo1.bin"), Path.Combine(release.RunFolder, "rvideo0.bin"), overwrite: true);
        VideoGenerationResult other = pipeline.Generate(request, settings, null, default);
        Assert.NotEqual(result.Frames.SelectMany(f => f.Rgb), other.Frames.SelectMany(f => f.Rgb));
        Assert.Throws<ArgumentException>(() => pipeline.Generate(request, settings with { Sampler = "euler" }, null, default));
    }

    /// <summary>The construction-time checks must read the distilled run folder as distilled.</summary>
    [Fact]
    public async Task AServiceBoundDistilledRequestPassesTheConstructionChecks()
    {
        using DuoReleaseFixture release = new(distilled: true);
        using InferenceEngine engine = new("cpu");
        ModelSpec spec = new() { Requested = "duo-v1-distill", Modality = Modality.Video, LocalPath = release.Release };
        VideoRequest request = Request((DuoRecipePipeline.RunFolderKey, release.RunFolder)) with { Steps = null };
        VideoPlan plan = await engine.VideoPlanning.PlanAsync(spec, request);

        using DuoRecipePipeline pipeline = new DuoRecipe(distilled: true).Build(
            new RecipeContext { CheckpointPath = release.Release, Backend = _backend, VideoPlan = plan },
            (_, _) => DuoReleaseFixture.Codecs());

        Assert.NotNull(pipeline);
    }

    [Fact]
    public async Task EachStageRefusesTheOtherStagesRunFolderAndRelease()
    {
        using DuoReleaseFixture release = new(distilled: true);
        DuoRecipe distilled = new(distilled: true);
        using DuoRecipePipeline causalPipeline = Build();
        VideoRequest renoised = Request((DuoRecipePipeline.RunFolderKey, release.RunFolder)) with { Steps = 4, FlowShift = 5f };
        VideoEffectiveSettings settings = (await Plan(renoised)).EffectiveSettings;
        ArgumentException refused = Assert.Throws<ArgumentException>(() => causalPipeline.Generate(renoised, settings, null, default));
        Assert.Contains("renoiseVideo", refused.Message, StringComparison.Ordinal);

        using DuoRecipePipeline distilledPipeline = distilled.Build(
            new RecipeContext { CheckpointPath = release.Release, Backend = _backend }, (_, _) => DuoReleaseFixture.Codecs());
        VideoRequest plain = Request() with { Steps = DuoReleaseFixture.Steps, FlowShift = 1f };
        VideoEffectiveSettings plainSettings = (await Plan(plain, distilled, "duo-v1-distill")).EffectiveSettings;
        refused = Assert.Throws<ArgumentException>(() => distilledPipeline.Generate(plain, plainSettings, null, default));
        Assert.Contains("renoiseVideo", refused.Message, StringComparison.Ordinal);

        Assert.Throws<NotSupportedException>(() => DuoAssets.Resolve(_release.Release, null, distilled: true));
        Assert.Throws<NotSupportedException>(() => DuoAssets.Resolve(release.Release, null));
    }

    /// <summary>A release folder named with <c>--model-path</c> holds the LoRA, so nothing is left to download.</summary>
    [Theory]
    [InlineData("duo-v1-causal")]
    [InlineData("duo-v1-2-causal")]
    [InlineData("duo-v1-distill")]
    public void AReleaseFolderSatisfiesTheCatalogFiles(string id)
    {
        CatalogEntry entry = ModelCatalog.All.Single(e => e.Id == id);

        Assert.Empty(ModelDownloader.MissingAssets(entry, _release.Release));
    }

    [Theory]
    [InlineData("duo-v1-causal")]
    [InlineData("duo-v1-distill")]
    public void ACataloguedRunIsRefusedUpFrontOnlyWhenItsReleaseFolderHasNoBackbone(string id)
    {
        CatalogEntry entry = ModelCatalog.All.Single(e => e.Id == id);
        ModelSpec Spec(string path) => new() { Requested = id, Modality = Modality.Video, Catalog = entry, LocalPath = path };
        // The release folder, its dcp folder and a file in it, the last two as they are named before any download.
        string[] present = [_release.Release, _release.Dcp, Path.Combine(_release.Dcp, "__0_0.distcp")];
        Assert.All(present, path => Assert.Null(DuoRecipe.BackboneRefusal(Spec(path))));

        string empty = Path.Combine(_release.Root, "empty");
        string[] absent = [empty, Path.Combine(empty, "realization", "dcp"), Path.Combine(empty, "realization", "dcp", "__0_0.distcp")];
        string expected = Path.Combine(empty, "realization", "backbone");
        Assert.All(absent, path => Assert.Contains(expected, DuoRecipe.BackboneRefusal(Spec(path)), StringComparison.Ordinal));
        Assert.Null(DuoRecipe.BackboneRefusal(new ModelSpec { Requested = empty, Modality = Modality.Video, LocalPath = empty }));
        // A LoRA folder on disk under another name is accepted, as construction accepts it.
        string renamed = Path.Combine(_release.Release, "realization", "lora");
        Directory.Move(_release.Dcp, renamed);
        Assert.Null(DuoRecipe.BackboneRefusal(Spec(renamed)));
        Assert.Null(DuoRecipe.BackboneRefusal(Spec(Path.Combine(renamed, "__0_0.distcp"))));
    }

    [Fact]
    public async Task TheDefaultsPlanWithoutIssuesAndAnOffGridFrameCountIsRefused()
    {
        Assert.Empty((await Plan(Request())).Issues);
        Assert.NotEmpty((await Plan(Request() with { Frames = 40 })).Issues);
        Assert.NotEmpty((await Plan(Request() with { InitImage = new ImageData { Rgb = new byte[12], Width = 2, Height = 2 } })).Issues);
    }

    [Fact]
    public async Task AGenerationReturnsEveryFrameAndTheSoundtrackOnTheVideoClock()
    {
        using DuoRecipePipeline pipeline = Build();
        VideoRequest request = Request();
        VideoEffectiveSettings settings = (await Plan(request)).EffectiveSettings;
        List<int> steps = [];

        VideoGenerationResult result = pipeline.Generate(request, settings, new Immediate(p => steps.Add(p.Step)), default);

        Assert.Equal(DuoReleaseFixture.Frames, result.Frames.Count);
        Assert.All(result.Frames, f => Assert.Equal((DuoReleaseFixture.Size, DuoReleaseFixture.Size), (f.Width, f.Height)));
        Assert.Equal(24, result.Fps);
        Assert.Equal(new[] { 1, 2, 3, 4 }, steps);
        AudioBuffer audio = result.Audio!;
        Assert.Equal(32000, audio.SampleRate);
        Assert.Equal(2, audio.ChannelCount);
        Assert.Equal(45334, audio.FrameCount);
        // Block 1 starts at sample 22667 and restarts at the first sample of its own decode; channel 1 keeps its own channel.
        Assert.Equal(22666 * 1e-6f, audio.Channels[0][22666] - audio.Channels[0][0], 5);
        float mean = audio.Channels[0][22667];
        Assert.Equal(1000 * 1e-6f, audio.Channels[0][23667] - mean, 5);
        Assert.Equal(0.25f, audio.Channels[1][22667] - mean, 5);
    }

    [Fact]
    public async Task ASecondGenerationOnTheSamePipelineRepeatsTheFirst()
    {
        using DuoRecipePipeline pipeline = Build();
        VideoRequest request = Request();
        VideoEffectiveSettings settings = (await Plan(request)).EffectiveSettings;

        VideoGenerationResult first = pipeline.Generate(request, settings, null, default);
        VideoGenerationResult second = pipeline.Generate(request, settings, null, default);

        Assert.Equal(first.Frames[^1].Rgb, second.Frames[^1].Rgb);
        Assert.Equal(first.Audio!.Channels[0], second.Audio!.Channels[0]);
    }

    /// <summary>The recipe merges the LoRA at load. A transformer built from the offline repack of the same release, driven by the same
    /// session, must give the same frames and audio.</summary>
    [Fact]
    public async Task TheLoadTimeMergeMatchesTheOfflineRepack()
    {
        using DuoRecipePipeline pipeline = Build();
        VideoRequest request = Request();
        VideoEffectiveSettings settings = (await Plan(request)).EffectiveSettings;
        VideoGenerationResult result = pipeline.Generate(request, settings, null, default);

        string repacked = Path.Combine(_release.Root, "repacked");
        using (DuoLoraSource lora = DuoLoraSource.FromDcp(_release.Dcp))
        {
            DuoRepacker.MergeBackbone(_release.Backbone, lora, repacked,
                new DuoMergeOptions { Heads = DuoReleaseFixture.Heads, HeadDim = DuoReleaseFixture.HeadDim });
        }
        using CheckpointSource source = CheckpointSource.OpenShards(HfCheckpointDirectory.WeightFiles(repacked));
        using MiniMaxH3Transformer transformer = MiniMaxH3Components.BuildTransformer(source, DType.F32,
            out MiniMaxH3Config config, out Dictionary<string, Tensor> weights);
        transformer.LoadWeights(weights);
        DuoRunInput input = DuoRunInput.Read(request, settings, config);
        DuoCodecs codecs = DuoReleaseFixture.Codecs();
        using DuoStreamCache cache = new(_backend, config.NumLayers, DuoReleaseFixture.Inner, input.CacheRows);
        using Diffusion.Models.Vae.DuoVideoStreamDecoder video = codecs.Video(_backend);
        using DuoAudioStreamDecoder audio = codecs.Audio(_backend);
        DuoOutput expected = new DuoSession(transformer, _backend, config, input.Sample, DuoRopePolicy.New, cache, video, audio)
            .Run(input.Run);

        byte[] expectedRgb = expected.Video.SelectMany(b => b.Rgb).ToArray();
        Assert.Equal(expectedRgb, result.Frames.SelectMany(f => f.Rgb).ToArray());
        Assert.Equal(DuoRecipePipeline.Soundtrack(expected.Audio).Channels[0], result.Audio!.Channels[0]);
    }

    [Fact]
    public async Task AShortWindowEvictsHistoryAndStillPublishesEveryBlock()
    {
        using DuoRecipePipeline pipeline = Build();
        VideoRequest request = Request((DuoRecipePipeline.WindowFramesKey, 34));
        VideoEffectiveSettings settings = (await Plan(request)).EffectiveSettings;

        // Two blocks of history room for one: the second commit evicts the first.
        Assert.Equal(4 + 3 + 78, DuoRunInput.Read(request, settings, Config()).CacheRows);
        Assert.Equal(DuoReleaseFixture.Frames, pipeline.Generate(request, settings, null, default).Frames.Count);
    }

    [Fact]
    public async Task ACancelledGenerationThrowsAndThePipelineStillRuns()
    {
        using DuoRecipePipeline pipeline = Build();
        VideoRequest request = Request();
        VideoEffectiveSettings settings = (await Plan(request)).EffectiveSettings;
        using CancellationTokenSource cancel = new();

        Assert.ThrowsAny<OperationCanceledException>(
            () => pipeline.Generate(request, settings, new Immediate(_ => cancel.Cancel()), cancel.Token));
        Assert.Equal(DuoReleaseFixture.Frames, pipeline.Generate(request, settings, null, default).Frames.Count);
    }

    [Fact]
    public async Task ARequestDuoCannotHonourIsRefused()
    {
        using DuoRecipePipeline pipeline = Build();
        VideoRequest request = Request();
        VideoEffectiveSettings settings = (await Plan(request)).EffectiveSettings;
        void Refused(VideoRequest r, VideoEffectiveSettings s) =>
            Assert.Throws<ArgumentException>(() => pipeline.Generate(r, s, null, default));

        Refused(request with { Prompt = "a face" }, settings);
        Refused(request with { Extra = new Dictionary<string, object>() }, settings);
        Refused(request with { TrimVideoStartFrames = 1 }, settings);
        Refused(request with { Fps = 30 }, settings);
        Refused(request, settings with { Steps = 3 });
        Refused(request, settings with { FlowShift = 5f });
        Refused(request, settings with { CfgScale = 2f });
        Refused(request, settings with { Sampler = "unipc" });
        Refused(request, settings with { Scheduler = "karras" });
        Refused(request, settings with { Frames = 51 });
        Refused(request, settings with { Width = 80 });
        Refused(Request((DuoRecipePipeline.WindowFramesKey, 17)), settings);
        // A history past the cached forward's row limit is refused before any work.
        Refused(request, settings with { Width = 2048, Height = 2048 });
    }

    /// <summary>The outside file holds 3 bytes, so a reader that sized it would complain about its rows and not its place.</summary>
    [Fact]
    public async Task AManifestNameOutsideTheRunFolderOrAMalformedManifestIsRefused()
    {
        using DuoRecipePipeline pipeline = Build();
        VideoRequest request = Request();
        VideoEffectiveSettings settings = (await Plan(request)).EffectiveSettings;
        string outside = Path.Combine(_release.Root, "outside.bin");
        File.WriteAllBytes(outside, [1, 2, 3]);
        string path = Path.Combine(_release.RunFolder, "manifest.json");
        string manifest = File.ReadAllText(path);
        string Refusal() => Assert.Throws<InvalidDataException>(() => pipeline.Generate(request, settings, null, default)).Message;

        foreach (string field in new[] { "shared.bin", "body1.bin" })
        {
            foreach (string name in new[] { "../outside.bin", outside.Replace('\\', '/') })
            {
                File.WriteAllText(path, manifest.Replace($"\"{field}\"", $"\"{name}\"", StringComparison.Ordinal));
                Assert.Contains("run folder", Refusal(), StringComparison.Ordinal);
            }
        }
        File.WriteAllText(path, "{ not json");
        Assert.Contains("not a run manifest", Refusal(), StringComparison.Ordinal);
        File.Delete(path);
        Assert.Contains("does not exist", Refusal(), StringComparison.Ordinal);
    }

    [Fact]
    public void TheReleaseResolvesFromItsFolderItsDcpFolderOrTheDistcpFile()
    {
        foreach (string path in new[] { _release.Release, _release.Dcp, Path.Combine(_release.Dcp, "__0_0.distcp") })
        {
            DuoAssets assets = DuoAssets.Resolve(path, null);
            Assert.Equal(_release.Dcp, assets.LoraDirectory);
            Assert.Equal(2, assets.BackboneShards.Count);
        }
    }

    [Fact]
    public void ADistilledStageARepackASingleFileAndAMissingCodecAreRefused()
    {
        string manifest = Path.Combine(_release.Release, "realization", "complete.json");
        File.WriteAllText(manifest, """{ "stage": "dmd" }""");
        Assert.Throws<NotSupportedException>(() => DuoAssets.Resolve(_release.Release, null));
        File.Delete(manifest);

        string repack = Path.Combine(_release.Backbone, DuoRepackManifest.FileName);
        File.WriteAllText(repack, "{}");
        Assert.Throws<NotSupportedException>(() => DuoAssets.Resolve(_release.Release, null));
        File.Delete(repack);

        File.Delete(Path.Combine(_release.Backbone, "model-00001-of-00002.safetensors"));
        File.Move(Path.Combine(_release.Backbone, "model-00002-of-00002.safetensors"),
            Path.Combine(_release.Backbone, "h3.safetensors"));
        Assert.Throws<NotSupportedException>(() => DuoAssets.Resolve(_release.Release, null));

        File.Copy(Path.Combine(_release.Backbone, "h3.safetensors"), Path.Combine(_release.Backbone, "h3-2.safetensors"));
        File.Delete(Path.Combine(_release.Release, "codec", "audio_vae", "diffusion_pytorch_model.safetensors"));
        Assert.Throws<FileNotFoundException>(() => DuoAssets.Resolve(_release.Release, null));
    }

    /// <summary>Without a stage manifest, the LoRA's key prefix still tells a distilled checkpoint apart.</summary>
    [Fact]
    public void ADistilledLoraWithoutAStageManifestIsRefusedAtConstruction()
    {
        File.Delete(Path.Combine(_release.Release, "realization", "complete.json"));
        DuoReleaseFixture.WriteLora(_release.Dcp, "models.generator.");

        InvalidDataException error = Assert.Throws<InvalidDataException>(() => Build());
        Assert.Contains("models.student.", error.Message, StringComparison.Ordinal);
    }

    /// <summary>The LoRA here is a distilled one, which only the merge reads: a request refusal that arrives instead of that error was made
    /// before the merge.</summary>
    [Fact]
    public async Task AServiceBoundRequestThatCannotRunIsRefusedAtConstructionBeforeTheMerge()
    {
        File.Delete(Path.Combine(_release.Release, "realization", "complete.json"));
        DuoReleaseFixture.WriteLora(_release.Dcp, "models.generator.");
        using InferenceEngine engine = new("cpu");
        ModelSpec spec = new() { Requested = "duo-v1-causal", Modality = Modality.Video, LocalPath = _release.Release };
        VideoPlan plan = await engine.VideoPlanning.PlanAsync(spec, Request() with { Prompt = "a face" });

        ArgumentException error = Assert.Throws<ArgumentException>(() => new DuoRecipe().Build(
            new RecipeContext { CheckpointPath = _release.Release, Backend = _backend, VideoPlan = plan },
            (_, _) => DuoReleaseFixture.Codecs()));

        Assert.Contains("no prompt", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AHistoryPastTheAvailableMemoryIsRefusedByTheWindowName()
    {
        DuoRunInput input = DuoRunInput.Read(Request(), (await Plan(Request())).EffectiveSettings, Config());
        MiniMaxH3Config config = Config() with
        {
            NumLayers = DuoReleaseFixture.Layers, NumAttentionHeads = DuoReleaseFixture.Heads,
            AttentionHeadDim = DuoReleaseFixture.HeadDim,
        };
        long history = input.HistoryBytes(config);
        Assert.Equal(input.CacheRows * 2L * 2 * 192 * 4, history);

        // Free memory short of the history is refused; the pool's unused reserve counts as available; no figure means no check.
        ArgumentException error = Assert.Throws<ArgumentException>(
            () => input.RequireHistoryFits(config, new DeviceMemoryUsage(history - 1, 2 * history, -1, -1, -1, -1)));
        Assert.Contains(DuoRecipePipeline.WindowFramesKey, error.Message, StringComparison.Ordinal);
        input.RequireHistoryFits(config, new DeviceMemoryUsage(history - 1, 2 * history, 10, 9, 0, 0));
        input.RequireHistoryFits(config, new DeviceMemoryUsage(0, 0, -1, -1, -1, -1));
    }

    [Fact]
    public void TheSoundtrackKeepsEachBlocksOwnSpanOfTheClock()
    {
        Assert.Equal(new long[] { 0, 22667, 45334, 68000 }, Enumerable.Range(0, 4).Select(DuoRecipePipeline.FirstSample).ToArray());
    }

    private DuoRecipePipeline Build() => new DuoRecipe().Build(
        new RecipeContext { CheckpointPath = _release.Release, Backend = _backend }, (_, _) => DuoReleaseFixture.Codecs());

    private VideoRequest Request(params (string Key, object Value)[] extra)
    {
        Dictionary<string, object> options = new() { [DuoRecipePipeline.RunFolderKey] = _release.RunFolder };
        foreach ((string key, object value) in extra)
        {
            options[key] = value;
        }
        return new VideoRequest
        {
            Prompt = "", Width = DuoReleaseFixture.Size, Height = DuoReleaseFixture.Size, Frames = DuoReleaseFixture.Frames,
            Steps = DuoReleaseFixture.Steps, Seed = 1, Extra = options,
        };
    }

    private static MiniMaxH3Config Config() => new()
    {
        TextDim = DuoReleaseFixture.TextDim, LatentsDim = 24, HasAudio = true, AudioLatentsDim = 32,
    };

    private static Task<VideoPlan> Plan(VideoRequest request) => Plan(request, new DuoRecipe(), "duo-v1-causal");

    private static Task<VideoPlan> Plan(VideoRequest request, DuoRecipe recipe, string id)
    {
        ModelSpec spec = new() { Requested = id, Modality = Modality.Video };
        return VideoProfileResolver.ResolveAsync(spec, request, VideoRecipeDeclaration.For(recipe, id, null, recipe.Supports), default);
    }

    private sealed class Immediate(Action<StepPreview> report) : IProgress<StepPreview>
    {
        public void Report(StepPreview value) => report(value);
    }
}
