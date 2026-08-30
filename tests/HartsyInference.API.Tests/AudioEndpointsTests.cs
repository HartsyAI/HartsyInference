using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>In-process tests for the native <c>/v1/native/{speech,transcribe,voice-convert,fx/*}</c> routes. Runs
/// on the CPU backend with no model ever loaded. Unlike images/text (which throw FileNotFoundException/
/// HartsyInferenceException for an unresolvable model → 400), every audio service resolves its catalog id through
/// an XCatalog.Resolve(id) call that throws NotSupportedException for an unknown id → 501 — confirmed by reading
/// SpeechService/TranscribeService/VoiceConversionService/FxService directly, not assumed.</summary>
public sealed class AudioEndpointsTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public AudioEndpointsTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder => builder.UseSetting("HartsyInference:Backend", "cpu"));
    }

    [Fact]
    public async Task Speech_UnknownModel_Returns501()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/speech", new
        {
            model = "not-a-real-tts-model",
            request = new { text = "hello" },
        });
        Assert.Equal(HttpStatusCode.NotImplemented, resp.StatusCode);
    }

    [Fact]
    public async Task Speech_MissingText_Returns400()
    {
        // SpeechService.SynthesizeAsync validates the text itself with ArgumentException, which GenerationErrors
        // maps to a caller error alongside the model-resolution exceptions.
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/speech", new
        {
            model = "kokoro",
            request = new { text = "" },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Transcribe_UnknownModel_Returns501()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/transcribe", new
        {
            model = "not-a-real-stt-model",
            request = new { audio = new { data = Convert.ToBase64String([1, 2, 3]) } },
        });
        Assert.Equal(HttpStatusCode.NotImplemented, resp.StatusCode);
    }

    [Fact]
    public async Task VoiceConvert_UnknownModel_Returns501()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/voice-convert", new
        {
            model = "not-a-real-vc-model",
            request = new { source = new { data = Convert.ToBase64String([1, 2, 3]) } },
        });
        Assert.Equal(HttpStatusCode.NotImplemented, resp.StatusCode);
    }

    [Fact]
    public async Task FxSeparate_GarbageAudioBytes_Returns400()
    {
        // Unlike Speech/Transcribe/VoiceConversion, FxService.SeparateAsync decodes the audio BEFORE resolving
        // the model id — there's no catalog-lookup-first step to hit for an "unknown model" case here. Garbage
        // bytes fail AudioClipCodec's decode with a HartsyInferenceException (→ 400), confirmed by reading
        // FxService/AudioClipCodec directly rather than assumed.
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/fx/separate", new
        {
            model = "not-a-real-fx-model",
            request = new { audio = new { data = Convert.ToBase64String([1, 2, 3]) } },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task FxEnhance_GarbageAudioBytes_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/fx/enhance", new
        {
            model = "not-a-real-fx-model",
            request = new { audio = new { data = Convert.ToBase64String([1, 2, 3]) } },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }
}
