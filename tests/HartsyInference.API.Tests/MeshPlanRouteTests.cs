using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;

namespace HartsyInference.API.Tests;

/// <summary>The mesh describe and plan routes over HTTP: the capability list, a typed 404, a plan that blocks, and a plan that passes without exposing a server path.</summary>
public sealed class MeshPlanRouteTests : IAsyncLifetime
{
    private WebApplication? _app;
    private HttpClient _http = null!;   // set in InitializeAsync

    public async Task InitializeAsync()
    {
        WebApplicationBuilder builder = WebApplication.CreateBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Services.AddHartsyInference();
        _app = builder.Build();
        _app.MapHartsyInferenceEndpoints();
        await _app.StartAsync();
        string address = _app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
        _http = new HttpClient { BaseAddress = new Uri(address), Timeout = TimeSpan.FromSeconds(30) };
    }

    public async Task DisposeAsync()
    {
        _http?.Dispose();
        if (_app is not null) await _app.DisposeAsync();
    }

    private static object Body(string model, object request, string? modelPath = null) => new
    {
        model,
        modelPath,
        request,
    };

    private static object Image() => new { rgb = Convert.ToBase64String(new byte[4 * 4 * 3]), width = 4, height = 4 };

    [Fact]
    public async Task DescribeListsEveryModelAndNamesTheirFields()
    {
        using JsonDocument all = JsonDocument.Parse(await _http.GetStringAsync("/v1/native/mesh/describe"));

        string[] ids = [.. all.RootElement.EnumerateArray().Select(e => e.GetProperty("id").GetString()!)];
        Assert.Contains("trellis2", ids);
        Assert.Contains("hairdeck", ids);
        JsonElement trellis2 = all.RootElement.EnumerateArray().Single(e => e.GetProperty("id").GetString() == "trellis2");
        Assert.Equal("Image", trellis2.GetProperty("input").GetString());
        Assert.Equal(3, trellis2.GetProperty("fields").GetProperty("GridResolution").GetProperty("values").GetArrayLength());
    }

    [Fact]
    public async Task DescribeOfAnUnknownModelIsA404()
    {
        HttpResponseMessage response = await _http.GetAsync("/v1/native/mesh/describe/hunyuan3d-mv");

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    [Fact]
    public async Task PlanOfAnUnsupportedFieldIs422WithTheIssue()
    {
        string folder = Path.GetTempPath();

        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/mesh/plan",
            Body("triposr", new { image = Image(), fieldOfViewRadians = 0.5 }, folder));

        Assert.Equal(HttpStatusCode.UnprocessableEntity, response.StatusCode);
        using JsonDocument plan = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.False(plan.RootElement.GetProperty("isValid").GetBoolean());
        JsonElement issue = plan.RootElement.GetProperty("issues").EnumerateArray().Single(i => i.GetProperty("code").GetString() == "mesh.request.unsupported");
        Assert.Equal("FieldOfViewRadians", issue.GetProperty("field").GetString());
    }

    [Fact]
    public async Task PlanOfAValidRequestIs200AndCarriesNoServerPath()
    {
        string folder = Path.GetTempPath();

        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/mesh/plan", Body("triposr", new { image = Image(), steps = 12 }, folder));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        string text = await response.Content.ReadAsStringAsync();
        Assert.DoesNotContain(folder.TrimEnd(Path.DirectorySeparatorChar), text, StringComparison.OrdinalIgnoreCase);
        using JsonDocument plan = JsonDocument.Parse(text);
        Assert.True(plan.RootElement.GetProperty("isValid").GetBoolean());
        Assert.Contains(plan.RootElement.GetProperty("issues").EnumerateArray(), i => i.GetProperty("code").GetString() == "mesh.field.ignored");
    }
}
