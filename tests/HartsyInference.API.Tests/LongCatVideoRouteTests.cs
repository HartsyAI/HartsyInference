using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using HartsyInference.Core.Configuration;
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

/// <summary>The native video routes over HTTP with the LongCat avatar recipe on a tiny release and the CPU: a portrait, a driving WAV and a chain
/// length arriving as JSON, planning, a three-segment generation through the video service, and the frames and soundtrack leaving as events. The
/// recipe is registered under a test family name with stand-ins for umT5, the VAE and the Whisper features, so no download or real weight is
/// touched.</summary>
[Collection(nameof(LongCatVideoRouteCollection))]
public sealed class LongCatVideoRouteTests : IAsyncLifetime
{
    private const string Family = "longcat-avatar-http-test";
    // 17 frames a segment and 23 in all take three segments.
    private const int Size = 16, SegmentFrames = 17, ChainFrames = 23, Steps = 8, Segments = 3;

    private readonly LongCatTinyAvatarRelease _release = new(onePointSix: false);
    private readonly LongCatRecipeStandIns _standIns = new(LongCatTinyAvatarRelease.Config.CaptionChannels,
        LongCatTinyAvatarRelease.Config.InChannels);
    private WebApplication? _app;
    private HttpClient _http = null!;   // set in InitializeAsync

    public async Task InitializeAsync()
    {
        VideoRecipeRegistry.Register(new TinyAvatarRecipe(new LongCatAvatarRecipe(_standIns.Sides, LongCatTinyAvatarRelease.Features(seed: 1))));
        KnobStore.Set(EngineKnobs.ModelsRoot, _release.Root);
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
        _standIns.Dispose();
        _release.Dispose();
    }

    [Fact]
    public async Task AnAvatarChainIsPlannedThenStreamedAsFramesAndSoundtrack()
    {
        HttpResponseMessage plan = await _http.PostAsJsonAsync("/v1/native/video/plan", Body());
        Assert.True(plan.StatusCode == HttpStatusCode.OK, $"{plan.StatusCode}: {await plan.Content.ReadAsStringAsync()}");

        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/video/stream", Body());
        string events = Encoding.UTF8.GetString(await response.Content.ReadAsByteArrayAsync());

        Assert.True(response.StatusCode == HttpStatusCode.OK, $"{response.StatusCode}: {events}");
        Assert.DoesNotContain("event: error", events, StringComparison.Ordinal);
        // The steps restart for each segment.
        Assert.Equal(Steps * Segments, Count(events, "event: progress"));
        Assert.Equal(ChainFrames, Count(events, "event: frame"));
        Assert.Equal(1, Count(events, "event: complete"));
        using JsonDocument complete = Data(events, "complete");
        Assert.Equal(ChainFrames, complete.RootElement.GetProperty("frames").GetInt32());
        using JsonDocument audio = Data(events, "audio");
        Assert.Equal(16_000, audio.RootElement.GetProperty("sampleRate").GetInt32());
        // The driving speech, cut to the clip: 23 frames at 25 fps of 16-bit mono behind the 44-byte WAV header.
        byte[] wav = Convert.FromBase64String(audio.RootElement.GetProperty("wav").GetString()!);
        Assert.Equal(ChainFrames * 16_000 / LongCatAvatarRecipe.OutputFps * 2, wav.Length - 44);
    }

    private object Body()
    {
        // Qualified: the API has an ImageData of its own, and the request carries the engine's.
        Engine.Requests.ImageData portrait = LongCatTinyAvatarRelease.Portrait(Size);
        AudioClip speech = LongCatTinyAvatarRelease.Speech(1.2);
        return new
        {
            model = Family,
            modelPath = _release.Checkpoint,
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

    private static int Count(string events, string name) => events.Split('\n').Count(line => line.TrimEnd('\r') == name);

    private static JsonDocument Data(string events, string name)
    {
        string[] lines = [.. events.Split('\n').Select(line => line.TrimEnd('\r'))];
        int at = Array.IndexOf(lines, "event: " + name);
        Assert.True(at >= 0, $"no {name} event");
        return JsonDocument.Parse(lines[at + 1]["data: ".Length..]);
    }

    /// <summary>The avatar recipe under a test family.</summary>
    private sealed class TinyAvatarRecipe(LongCatAvatarRecipe inner) : IVideoRecipe
    {
        private readonly IVideoRecipe _inner = inner;

        public string Name => Family;

        public PromptProfile? Prompt => _inner.Prompt;

        public VideoDefaults Defaults => _inner.Defaults;

        public VideoConstraints Constraints => _inner.Constraints;

        public VideoFeatures Supports => _inner.Supports;

        public MemoryCapabilities MemorySupports => _inner.MemorySupports;

        public bool RefusesUnhonouredPlacements => _inner.RefusesUnhonouredPlacements;

        public bool Matches(string familyId) => string.Equals(familyId, Family, StringComparison.OrdinalIgnoreCase);

        public IVideoRecipePipeline Construct(RecipeContext context) => _inner.Construct(context);
    }
}

/// <summary>Runs alone: the test sets the process-wide models root.</summary>
[CollectionDefinition(nameof(LongCatVideoRouteCollection), DisableParallelization = true)]
public sealed class LongCatVideoRouteCollection
{
}
