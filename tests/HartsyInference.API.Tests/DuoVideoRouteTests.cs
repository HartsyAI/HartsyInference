using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using HartsyInference.Core.Configuration;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Tests.Duo;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Metadata;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;

namespace HartsyInference.API.Tests;

/// <summary>The native video routes over HTTP with the Duo recipe on a synthetic release and the CPU: both <c>Extra</c> keys arriving as
/// JSON, planning and generation through the video service, and the run-folder rule. The recipe is registered under a test family name and
/// builds fake codec decoders, so the public <c>Construct</c> and the codec loaders do not run here.</summary>
[Collection(nameof(DuoVideoRouteCollection))]
public sealed class DuoVideoRouteTests : IAsyncLifetime
{
    private const string Family = "duo-http-test";

    private readonly DuoReleaseFixture _release = new();
    private WebApplication? _app;
    private HttpClient _http = null!;   // set in InitializeAsync

    public async Task InitializeAsync()
    {
        if (VideoRecipeRegistry.Resolve(Family) is null)
        {
            VideoRecipeRegistry.Register(new FakeCodecDuoRecipe());
        }
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
        _release.Dispose();
    }

    [Fact]
    public async Task TheStreamRouteGeneratesFromARunFolderUnderTheModelsRoot()
    {
        string events = await Stream(Body("run", window: 34));

        Assert.DoesNotContain("event: error", events, StringComparison.Ordinal);
        Assert.Equal(DuoReleaseFixture.Frames, Count(events, "event: frame"));
        Assert.Equal(1, Count(events, "event: complete"));
        using JsonDocument audio = Data(events, "audio");
        Assert.Equal(32000, audio.RootElement.GetProperty("sampleRate").GetInt32());
        Assert.Equal(2, audio.RootElement.GetProperty("channels").GetInt32());
        // All 45334 samples of 34 frames at 24 fps survive the service's fit to the clip, in a 16-bit stereo WAV.
        byte[] wav = Convert.FromBase64String(audio.RootElement.GetProperty("wav").GetString()!);
        Assert.Equal(45334 * 2 * 2, wav.Length - 44);
    }

    [Fact]
    public async Task AWindowTheRecipeRefusesEndsTheStreamWithItsReason()
    {
        string events = await Stream(Body("run", window: 17));

        Assert.Contains("event: error", events, StringComparison.Ordinal);
        Assert.Contains(DuoRecipePipeline.WindowFramesKey, events, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AGenerationErrorNamingAServerPathReachesTheClientWithoutIt()
    {
        File.Delete(Path.Combine(_release.RunFolder, "manifest.json"));

        string events = await Stream(Body("run", window: 34));

        Assert.Contains("local artifact details were omitted", events, StringComparison.Ordinal);
        Assert.DoesNotContain(Path.GetFileName(_release.Root), events, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("/v1/native/video/plan")]
    [InlineData("/v1/native/video/stream")]
    public async Task ARunFolderOutsideTheModelsRootOrNotANameIsA400(string route)
    {
        foreach (object folder in new object[] { "..", "../elsewhere", Path.GetTempPath(), 7, "" })
        {
            HttpResponseMessage response = await _http.PostAsJsonAsync(route, Body(folder, window: 34));

            Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
            Assert.Contains(DuoRecipePipeline.RunFolderKey, await response.Content.ReadAsStringAsync(), StringComparison.Ordinal);
        }
    }

    [Theory]
    [InlineData("/v1/native/video/plan")]
    [InlineData("/v1/native/video/stream")]
    public async Task AMissingRunFolderIsA404NamingWhatTheClientSent(string route)
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync(route, Body("no-such-run", window: 34));
        string text = await response.Content.ReadAsStringAsync();

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        Assert.Contains("no-such-run", text, StringComparison.Ordinal);
        Assert.DoesNotContain(Path.GetFileName(_release.Root), text, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ThePlanRouteAcceptsAFullPathUnderTheModelsRoot()
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/video/plan", Body(_release.RunFolder, window: 34));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Theory]
    [InlineData("/v1/native/video/plan")]
    [InlineData("/v1/native/video/stream")]
    public async Task APromptIsA422FromPlanning(string route)
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync(route, Body("run", window: 34, prompt: "a face"));

        Assert.Equal(HttpStatusCode.UnprocessableEntity, response.StatusCode);
        Assert.Contains("video.request.unsupported", await response.Content.ReadAsStringAsync(), StringComparison.Ordinal);
    }

    private object Body(object folder, int window, string prompt = "") => new
    {
        model = Family,
        modelPath = _release.Release,
        save = false,
        request = new
        {
            prompt,
            width = DuoReleaseFixture.Size,
            height = DuoReleaseFixture.Size,
            frames = DuoReleaseFixture.Frames,
            steps = DuoReleaseFixture.Steps,
            seed = 1,
            extra = new Dictionary<string, object>
            {
                [DuoRecipePipeline.RunFolderKey] = folder,
                [DuoRecipePipeline.WindowFramesKey] = window,
            },
        },
    };

    private async Task<string> Stream(object body)
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/video/stream", body);
        string text = Encoding.UTF8.GetString(await response.Content.ReadAsByteArrayAsync());
        Assert.True(response.StatusCode == HttpStatusCode.OK, $"{response.StatusCode}: {text}");
        return text;
    }

    private static int Count(string events, string name) => events.Split('\n').Count(line => line.TrimEnd('\r') == name);

    private static JsonDocument Data(string events, string name)
    {
        string[] lines = [.. events.Split('\n').Select(line => line.TrimEnd('\r'))];
        int at = Array.IndexOf(lines, "event: " + name);
        Assert.True(at >= 0, $"no {name} event");
        return JsonDocument.Parse(lines[at + 1]["data: ".Length..]);
    }

    /// <summary>The Duo recipe under a test family, building the fixture's fake codec decoders.</summary>
    private sealed class FakeCodecDuoRecipe : IVideoRecipe
    {
        private readonly DuoRecipe _duo = new();

        public string Name => Family;

        public PromptProfile? Prompt => _duo.Prompt;

        public VideoDefaults Defaults => _duo.Defaults;

        public VideoConstraints Constraints => _duo.Constraints;

        public VideoFeatures Supports => _duo.Supports;

        public bool RefusesUnhonouredPlacements => _duo.RefusesUnhonouredPlacements;

        public string? RequestRefusal(VideoRequest request, VideoEffectiveSettings settings) => _duo.RequestRefusal(request, settings);

        public bool Matches(string familyId) => string.Equals(familyId, Family, StringComparison.OrdinalIgnoreCase);

        public SamplingCapabilities.SamplingSupport SamplingSupportFor(string familyId, ResolvedModelVariant? variant) =>
            _duo.SamplingSupportFor(familyId, variant);

        public IVideoRecipePipeline Construct(RecipeContext context) =>
            _duo.Build(context, (_, _) => DuoReleaseFixture.Codecs());
    }
}

/// <summary>Runs alone: the test sets the process-wide models root.</summary>
[CollectionDefinition(nameof(DuoVideoRouteCollection), DisableParallelization = true)]
public sealed class DuoVideoRouteCollection
{
}
