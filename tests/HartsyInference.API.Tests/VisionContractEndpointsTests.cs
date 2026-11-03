using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using HartsyInference.API.Endpoints;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>The P2 vision contracts on the HTTP surface: the validator refuses malformed requests with 400 before a model
/// loads, a valid request for a mode no model implements is a 501 (never an SSE stream that errors later), and structured
/// results travel as float32 arrays whose bytes are the CLI's <c>.npy</c> payload.</summary>
public sealed class VisionContractEndpointsTests : IClassFixture<WebApplicationFactory<Program>>
{
    private static readonly JsonSerializerOptions Web = new(JsonSerializerDefaults.Web) { Converters = { new JsonStringEnumConverter() } };
    private readonly WebApplicationFactory<Program> _factory;

    public VisionContractEndpointsTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder => builder.UseSetting("HartsyInference:Backend", "cpu"));
    }

    [Fact]
    public async Task Vision_SequenceOnlyModeOnASingleImage_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/vision", new { model = "whatever", request = Single("Track") });

        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        Assert.Contains("needs a frame sequence", await ErrorMessage(resp), StringComparison.Ordinal);
    }

    [Fact]
    public async Task Vision_NewSingleImageModeWithNoModel_Returns501()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/vision", new { model = "whatever", request = Single("Pose") });

        Assert.Equal(HttpStatusCode.NotImplemented, resp.StatusCode);
        Assert.Contains("No Pose model", await ErrorMessage(resp), StringComparison.Ordinal);
    }

    [Fact]
    public async Task Sequence_TrackWithoutPrompts_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/vision/sequence", new
        {
            model = "whatever",
            request = new { images = Images(3), mode = "Track", kind = "Frames", fps = 24.0 },
        });

        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        Assert.Contains("at least one prompt", await ErrorMessage(resp), StringComparison.Ordinal);
    }

    [Fact]
    public async Task Sequence_ValidFramesWithNoModel_Returns501BeforeAnyStream()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/vision/sequence", new
        {
            model = "whatever",
            request = new { images = Images(3), mode = "Interpolate", kind = "Frames", fps = 24.0, interpolationFactor = 2 },
        });

        Assert.Equal(HttpStatusCode.NotImplemented, resp.StatusCode);
        Assert.NotEqual("text/event-stream", resp.Content.Headers.ContentType?.MediaType);
    }

    [Fact]
    public async Task Sequence_ValidViewsWithNoModel_Returns501()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/vision/sequence", new
        {
            model = "whatever",
            request = new { images = Images(2), mode = "Geometry", kind = "Views" },
        });

        Assert.Equal(HttpStatusCode.NotImplemented, resp.StatusCode);
        Assert.Contains("multi-view", await ErrorMessage(resp), StringComparison.Ordinal);
    }

    [Fact]
    public void Wire_ArraysCarryTheNpyPayloadAndTheConventions()
    {
        VisionArray points = new VisionArray([0.5f, -1f, 2f, 3.25f, 4f, -5.5f], 1, 2, 3);
        VisionResult result = new VisionResult
        {
            DepthMap = new VisionDepthMap { Values = new VisionArray([1.5f, 2.5f], 1, 2), Kind = VisionDepthKind.MetricMeters },
            PointMap = points,
            Camera = new VisionCamera { Fx = 500, Fy = 500, Cx = 1, Cy = 0.5, Width = 2, Height = 1 },
            Views = [new VisionResult { Normals = new VisionArray([0f, 0f, 1f, 0f, 1f, 0f], 1, 2, 3) }],
        };

        JsonElement json = JsonSerializer.SerializeToElement(VisionWire.Result(result), Web);

        JsonElement wirePoints = json.GetProperty("pointMap");
        Assert.Equal("float32", wirePoints.GetProperty("dtype").GetString());
        Assert.Equal([1, 2, 3], wirePoints.GetProperty("shape").EnumerateArray().Select(d => d.GetInt32()));
        byte[] payload = Convert.FromBase64String(wirePoints.GetProperty("data").GetString()!);
        byte[] npy = VisionArrayCodec.ToNpy(points);
        // The CLI's .npy carries exactly these bytes after its header.
        Assert.Equal(payload, npy[^payload.Length..]);
        Assert.Equal(points.Data, VisionArrayCodec.FromLittleEndianBytes(payload, points.Shape).Data);

        Assert.Equal("MetricMeters", json.GetProperty("depthMap").GetProperty("kind").GetString());
        Assert.Equal(500, json.GetProperty("camera").GetProperty("fx").GetDouble());
        Assert.Equal(0.5, json.GetProperty("conventions").GetProperty("pixelCenter").GetDouble());
        JsonElement view = json.GetProperty("views")[0];
        Assert.Equal([1, 2, 3], view.GetProperty("normals").GetProperty("shape").EnumerateArray().Select(d => d.GetInt32()));
    }

    [Fact]
    public void Wire_PlainResultsCarryNoConventions()
    {
        JsonElement json = JsonSerializer.SerializeToElement(VisionWire.Result(new VisionResult { Embedding = [1f, 2f] }), Web);

        Assert.Equal(JsonValueKind.Null, json.GetProperty("conventions").ValueKind);
        Assert.Equal(2, json.GetProperty("embedding").GetArrayLength());
    }

    private static object Single(string mode) => new { image = Image(), mode };

    private static object Image() => new { rgb = Convert.ToBase64String(new byte[4 * 3 * 3]), width = 4, height = 3 };

    private static object[] Images(int count) => Enumerable.Range(0, count).Select(_ => Image()).ToArray();

    private static async Task<string> ErrorMessage(HttpResponseMessage resp)
    {
        JsonElement body = await resp.Content.ReadFromJsonAsync<JsonElement>();
        return body.GetProperty("error").GetProperty("message").GetString()!;
    }
}
