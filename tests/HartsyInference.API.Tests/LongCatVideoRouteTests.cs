using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Tests;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.Metadata;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;

namespace HartsyInference.API.Tests;

/// <summary>The native video routes over HTTP with the LongCat recipes on tiny releases and the CPU: the base recipe from text and from a first
/// frame, and the avatar recipe's 1.5 and 1.6 builds running a three-segment chain from a portrait and a driving WAV. The recipes are registered
/// under test family names with stand-ins for umT5, the VAE and the Whisper features, so no download or real weight is touched. The LoRA refusal
/// is checked on the production family ids, which plan without loading a model.</summary>
// The models root is process-global: this class shares the "ModelsRootKnob" collection with every other test that sets it, so they never overlap.
[Collection("ModelsRootKnob")]
public sealed class LongCatVideoRouteTests : IAsyncLifetime
{
    private const string AvatarFamily = "longcat-avatar-http-test", BaseFamily = "longcat-video-http-test";
    // The avatar: 17 frames a segment and 23 in all take three segments.
    private const int Size = 16, SegmentFrames = 17, ChainFrames = 23, Steps = 8, Segments = 3;
    private const int BaseFrames = 5, BaseSteps = 2;

    private readonly string _root = Path.Combine(Path.GetTempPath(), "longcat-route-" + Guid.NewGuid().ToString("N"));
    private readonly LongCatTinyAvatarRelease _avatar15 = new(onePointSix: false);
    private readonly LongCatTinyAvatarRelease _avatar16 = new(onePointSix: true);
    private readonly LongCatRecipeStandIns _avatarStandIns = new(LongCatTinyAvatarRelease.Config.CaptionChannels,
        LongCatTinyAvatarRelease.Config.InChannels);
    private readonly LongCatRecipeStandIns _baseStandIns = new(LongCatTestData.Tiny.CaptionChannels, LongCatTestData.Tiny.InChannels);
    private WebApplication? _app;
    private HttpClient _http = null!;   // set in InitializeAsync

    private string BaseCheckpoint => Path.Combine(_root, "dit");

    public async Task InitializeAsync()
    {
        Directory.CreateDirectory(BaseCheckpoint);
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(LongCatTestData.Tiny, seed: 71);
        LongCatTestData.WriteCheckpoint(BaseCheckpoint, weights);
        LongCatTestData.DisposeAll(weights);
        VideoRecipeRegistry.Register(new UnderFamily(AvatarFamily,
            new LongCatAvatarRecipe(_avatarStandIns.Sides, LongCatTinyAvatarRelease.Features(seed: 1))));
        VideoRecipeRegistry.Register(new UnderFamily(BaseFamily, new LongCatVideoRecipe(_baseStandIns.Sides)));
        KnobStore.Set(EngineKnobs.ModelsRoot, _root);
        WebApplicationBuilder builder = WebApplication.CreateBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Services.AddHartsyInference(options => options.Backend = "cpu");
        _app = builder.Build();
        _app.MapHartsyInferenceEndpoints();
        await _app.StartAsync();
        string address = _app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
        _http = new HttpClient { BaseAddress = new Uri(address), Timeout = TimeSpan.FromSeconds(120) };
    }

    public async Task DisposeAsync()
    {
        _http?.Dispose();
        if (_app is not null) await _app.DisposeAsync();
        KnobStore.Clear(EngineKnobs.ModelsRoot);
        _avatarStandIns.Dispose();
        _baseStandIns.Dispose();
        _avatar15.Dispose();
        _avatar16.Dispose();
        Directory.Delete(_root, recursive: true);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task AnAvatarChainIsPlannedThenStreamedAsFramesAndSoundtrack(bool onePointSix)
    {
        // The 1.6 release is refused by the 1.5 build, so a run of it completing shows the variant reached the recipe.
        string events = await PlanThenStream(AvatarBody(onePointSix));

        // The steps restart for each segment.
        Assert.Equal(Steps * Segments, Count(events, "event: progress"));
        AssertFrames(events, ChainFrames);
        using JsonDocument audio = Data(events, "audio");
        Assert.Equal(16_000, audio.RootElement.GetProperty("sampleRate").GetInt32());
        // The driving speech, cut to the clip: 23 frames at 25 fps of 16-bit mono behind the 44-byte WAV header.
        byte[] wav = Convert.FromBase64String(audio.RootElement.GetProperty("wav").GetString()!);
        Assert.Equal(ChainFrames * 16_000 / LongCatAvatarRecipe.OutputFps * 2, wav.Length - 44);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ABaseClipIsPlannedThenStreamedFromTextOrAFirstFrame(bool fromImage)
    {
        Engine.Requests.ImageData? first = fromImage ? LongCatTinyAvatarRelease.Portrait(Size) : null;
        object body = new
        {
            model = BaseFamily,
            modelPath = BaseCheckpoint,
            save = false,
            request = new
            {
                prompt = "a lighthouse at dusk",
                width = Size,
                height = Size,
                frames = BaseFrames,
                steps = BaseSteps,
                cfgScale = 1.0,
                seed = 3,
                initImage = first is null ? null : new { rgb = first.Rgb, width = first.Width, height = first.Height },
            },
        };

        string events = await PlanThenStream(body);

        Assert.Equal(BaseSteps, Count(events, "event: progress"));
        AssertFrames(events, BaseFrames);
        Assert.Equal(0, Count(events, "event: audio"));
        // The first frame is the only thing the base recipe encodes.
        Assert.Equal(fromImage ? [1] : [], _baseStandIns.Vae.EncodedClipFrames);
    }

    [Theory]
    [InlineData("longcat-video", "/v1/native/video/plan")]
    [InlineData("longcat-video", "/v1/native/video/stream")]
    [InlineData("longcat-avatar", "/v1/native/video/plan")]
    [InlineData("longcat-avatar", "/v1/native/video/stream")]
    [InlineData("longcat-avatar-1.6", "/v1/native/video/plan")]
    [InlineData("longcat-avatar-1.6", "/v1/native/video/stream")]
    public async Task AUserLoraIsRefusedWithItsReason(string model, string route)
    {
        string checkpoint = model switch
        {
            "longcat-video" => BaseCheckpoint,
            "longcat-avatar" => _avatar15.Checkpoint,
            _ => _avatar16.Checkpoint,
        };
        object body = new
        {
            model,
            modelPath = checkpoint,
            save = false,
            request = new { prompt = "a person speaking", loras = new { entries = new[] { new { model = "some-lora", weight = 1.0 } } } },
        };

        HttpResponseMessage response = await _http.PostAsJsonAsync(route, body);
        string text = await response.Content.ReadAsStringAsync();

        Assert.True(response.StatusCode == HttpStatusCode.UnprocessableEntity, $"{response.StatusCode}: {text}");
        Assert.Contains("video.feature.unsupported", text, StringComparison.Ordinal);
        Assert.Contains("does not support: Lora", text, StringComparison.Ordinal);
    }

    /// <summary>A continuation request reaches VideoRequest.VideoContinuation through the JSON route, so the base family plans it; the planner refuses
    /// it for a family that does not declare the feature.</summary>
    [Fact]
    public async Task AContinuationRequestReachesTheBaseFamilyAndIsRefusedByTheAvatar()
    {
        object clip = new { data = Convert.ToBase64String(new byte[] { 1, 2, 3 }), format = "mp4" };
        object baseBody = new
        {
            model = BaseFamily,
            modelPath = BaseCheckpoint,
            save = false,
            request = new { prompt = "the road continues", frames = BaseFrames, steps = BaseSteps, cfgScale = 1.0, seed = 3, videoContinuation = clip },
        };

        HttpResponseMessage plan = await _http.PostAsJsonAsync("/v1/native/video/plan", baseBody);
        string planText = await plan.Content.ReadAsStringAsync();
        Assert.True(plan.StatusCode == HttpStatusCode.OK, $"{plan.StatusCode}: {planText}");

        object avatarBody = new
        {
            model = "longcat-avatar",
            modelPath = _avatar15.Checkpoint,
            save = false,
            request = new { prompt = "the speaker continues", videoContinuation = clip },
        };
        HttpResponseMessage refused = await _http.PostAsJsonAsync("/v1/native/video/plan", avatarBody);
        string refusedText = await refused.Content.ReadAsStringAsync();
        Assert.True(refused.StatusCode == HttpStatusCode.UnprocessableEntity, $"{refused.StatusCode}: {refusedText}");
        Assert.Contains("does not support: VideoContinuation", refusedText, StringComparison.Ordinal);
    }

    private object AvatarBody(bool onePointSix)
    {
        Engine.Requests.ImageData portrait = LongCatTinyAvatarRelease.Portrait(Size);
        AudioClip speech = LongCatTinyAvatarRelease.Speech(1.2);
        return new
        {
            model = onePointSix ? AvatarFamily + ":1.6" : AvatarFamily,
            modelPath = (onePointSix ? _avatar16 : _avatar15).Checkpoint,
            save = false,
            request = new
            {
                prompt = "a person speaking",
                width = Size,
                height = Size,
                frames = SegmentFrames,
                fps = LongCatAvatarRecipe.OutputFps,
                seed = 3,
                chainTotalFrames = ChainFrames,
                initImage = new { rgb = portrait.Rgb, width = portrait.Width, height = portrait.Height },
                videoAudioReference = new { data = speech.Data, format = speech.Format },
            },
        };
    }

    private async Task<string> PlanThenStream(object body)
    {
        HttpResponseMessage plan = await _http.PostAsJsonAsync("/v1/native/video/plan", body);
        Assert.True(plan.StatusCode == HttpStatusCode.OK, $"{plan.StatusCode}: {await plan.Content.ReadAsStringAsync()}");

        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/video/stream", body);
        string events = Encoding.UTF8.GetString(await response.Content.ReadAsByteArrayAsync());

        Assert.True(response.StatusCode == HttpStatusCode.OK, $"{response.StatusCode}: {events}");
        Assert.DoesNotContain("event: error", events, StringComparison.Ordinal);
        return events;
    }

    private static void AssertFrames(string events, int frames)
    {
        Assert.Equal(frames, Count(events, "event: frame"));
        Assert.Equal(1, Count(events, "event: complete"));
        using JsonDocument complete = Data(events, "complete");
        Assert.Equal(frames, complete.RootElement.GetProperty("frames").GetInt32());
    }

    private static int Count(string events, string name) => events.Split('\n').Count(line => line.TrimEnd('\r') == name);

    private static JsonDocument Data(string events, string name)
    {
        string[] lines = [.. events.Split('\n').Select(line => line.TrimEnd('\r'))];
        int at = Array.IndexOf(lines, "event: " + name);
        Assert.True(at >= 0, $"no {name} event");
        return JsonDocument.Parse(lines[at + 1]["data: ".Length..]);
    }

    /// <summary>A recipe under a test family name.</summary>
    private sealed class UnderFamily(string family, IVideoRecipe inner) : IVideoRecipe
    {
        public string Name => family;

        public PromptProfile? Prompt => inner.Prompt;

        public VideoDefaults Defaults => inner.Defaults;

        public VideoConstraints Constraints => inner.Constraints;

        public VideoFeatures Supports => inner.Supports;

        public MemoryCapabilities MemorySupports => inner.MemorySupports;

        public bool RefusesUnhonouredPlacements => inner.RefusesUnhonouredPlacements;

        public bool Matches(string familyId) => string.Equals(familyId, family, StringComparison.OrdinalIgnoreCase);

        public IVideoRecipePipeline Construct(RecipeContext context) => inner.Construct(context);
    }
}
