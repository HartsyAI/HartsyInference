using System.ComponentModel;
using System.Text.Json;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Tools.Tests;

/// <summary><see cref="ToolSchema.FromDelegate"/>'s parameter-to-schema mapping and the delegate binding behind <see cref="ToolRegistry.Add(string, Delegate, string?)"/>: descriptions, required vs optional, enums, conversions, async results, and the error results an unknown or failing tool yields.</summary>
public sealed class ToolSchemaTests
{
    private enum Unit
    {
        Celsius,
        Fahrenheit,
    }

    [Description("Looks up the forecast for a city.")]
    private static string Forecast(
        [Description("City name, e.g. Paris")] string city,
        [Description("Days ahead")] int days = 3,
        double? latitude = null,
        bool metric = false,
        Unit unit = Unit.Celsius,
        CancellationToken cancel = default)
        => $"{city}/{days}/{latitude?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "-"}/{metric}/{unit}/{cancel.CanBeCanceled}";

    [Fact]
    public void SchemaMapsParametersTypesDescriptionsAndRequired()
    {
        ToolDefinition definition = ToolSchema.FromDelegate("forecast", Forecast);
        Assert.Equal("forecast", definition.Name);
        Assert.Equal("Looks up the forecast for a city.", definition.Description);
        using JsonDocument doc = JsonDocument.Parse(definition.JsonSchema);
        JsonElement root = doc.RootElement;
        Assert.Equal("object", root.GetProperty("type").GetString());
        JsonElement properties = root.GetProperty("properties");
        Assert.Equal("string", properties.GetProperty("city").GetProperty("type").GetString());
        Assert.Equal("City name, e.g. Paris", properties.GetProperty("city").GetProperty("description").GetString());
        Assert.Equal("integer", properties.GetProperty("days").GetProperty("type").GetString());
        Assert.Equal("Days ahead", properties.GetProperty("days").GetProperty("description").GetString());
        Assert.Equal("number", properties.GetProperty("latitude").GetProperty("type").GetString());
        Assert.Equal("boolean", properties.GetProperty("metric").GetProperty("type").GetString());
        Assert.Equal("string", properties.GetProperty("unit").GetProperty("type").GetString());
        Assert.Equal(["Celsius", "Fahrenheit"], properties.GetProperty("unit").GetProperty("enum").EnumerateArray().Select(e => e.GetString()));
        Assert.False(properties.TryGetProperty("cancel", out _));
        Assert.Equal(["city"], root.GetProperty("required").EnumerateArray().Select(e => e.GetString()));
    }

    [Fact]
    public void ExplicitDescriptionWinsAndNoArgumentToolHasEmptyProperties()
    {
        ToolDefinition definition = ToolSchema.FromDelegate("hang_up", () => "ok", "Ends the call.");
        Assert.Equal("Ends the call.", definition.Description);
        using JsonDocument doc = JsonDocument.Parse(definition.JsonSchema);
        Assert.Empty(doc.RootElement.GetProperty("properties").EnumerateObject());
        Assert.Equal(0, doc.RootElement.GetProperty("required").GetArrayLength());
    }

    [Fact]
    public void UnsupportedParameterTypeFailsAtRegistration()
        => Assert.Throws<NotSupportedException>(() => ToolSchema.FromDelegate("bad", (List<int> xs) => xs.Count));

    [Fact]
    public async Task DelegateBindingConvertsArgumentsAndFillsDefaults()
    {
        ToolRegistry registry = new ToolRegistry().Add("forecast", Forecast);
        Assert.Equal(["forecast"], registry.Names);
        string full = await registry.InvokeAsync(new NativeToolCall
        {
            Name = "forecast",
            Arguments = "{\"City\": \"Paris\", \"days\": \"5\", \"latitude\": 48.8, \"metric\": true, \"unit\": \"fahrenheit\"}",
        });
        Assert.Equal("Paris/5/48.8/True/Fahrenheit/False", full);
        string defaults = await registry.InvokeAsync(new NativeToolCall { Name = "forecast", Arguments = "{\"city\": \"Oslo\"}" });
        Assert.Equal("Oslo/3/-/False/Celsius/False", defaults);
        string empty = await registry.InvokeAsync(new NativeToolCall { Name = "forecast", Arguments = "" });
        Assert.StartsWith("{\"error\":", empty, StringComparison.Ordinal);
        Assert.Contains("city", empty, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AsyncAndNonStringResultsAreRendered()
    {
        ToolRegistry registry = new ToolRegistry()
            .Add("async_text", async (int n, CancellationToken cancel) =>
            {
                await Task.Delay(1, cancel);
                return $"n={n}";
            })
            .Add("number", (double x) => x * 2)
            .Add("void", static () => Task.CompletedTask);
        Assert.Equal("n=7", await registry.InvokeAsync(new NativeToolCall { Name = "async_text", Arguments = "{\"n\": 7}" }));
        Assert.Equal("5", await registry.InvokeAsync(new NativeToolCall { Name = "number", Arguments = "{\"x\": 2.5}" }));
        Assert.Equal("", await registry.InvokeAsync(new NativeToolCall { Name = "void", Arguments = "{}" }));
    }

    [Fact]
    public async Task UnknownToolAndThrowingHandlerYieldErrorResults()
    {
        ToolRegistry registry = new ToolRegistry().Add("boom", static string () => throw new InvalidOperationException("kaboom"));
        string unknown = await registry.InvokeAsync(new NativeToolCall { Name = "nope", Arguments = "{}" });
        Assert.Equal("{\"error\":\"Unknown tool 'nope'. Available tools: boom.\"}", unknown);
        string thrown = await registry.InvokeAsync(new NativeToolCall { Name = "boom", Arguments = "{}" });
        Assert.Equal("{\"error\":\"InvalidOperationException: kaboom\"}", thrown);
    }

    [Fact]
    public async Task CancellationPropagatesOutOfDispatch()
    {
        ToolRegistry registry = new ToolRegistry().Add("wait", (CancellationToken cancel) => Task.Delay(Timeout.Infinite, cancel));
        using CancellationTokenSource cts = new();
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => registry.InvokeAsync(new NativeToolCall { Name = "wait", Arguments = "{}" }, cts.Token));
    }

    [Fact]
    public void DuplicateNamesAreRejectedAndCustomHandlersKeepTheirSchema()
    {
        ToolRegistry registry = new ToolRegistry().Add("t", "desc", "{\"type\":\"object\"}", static (_, _) => Task.FromResult("r"));
        Assert.Throws<ArgumentException>(() => registry.Add("t", () => "again"));
        ToolDefinition definition = Assert.Single(registry.Definitions);
        Assert.Equal("desc", definition.Description);
        Assert.Equal("{\"type\":\"object\"}", definition.JsonSchema);
        Assert.True(registry.TryGet("t", out IToolHandler handler));
        Assert.Equal("t", handler.Name);
    }


    [Fact]
    public async Task ValueTaskResultsAreUnwrapped()
    {
        ToolRegistry registry = new ToolRegistry()
            .Add("count", (int n) => new ValueTask<int>(n + 1))
            .Add("nothing", static () => ValueTask.CompletedTask);
        Assert.Equal("5", await registry.InvokeAsync(new NativeToolCall { Name = "count", Arguments = "{\"n\": 4}" }));
        Assert.Equal("", await registry.InvokeAsync(new NativeToolCall { Name = "nothing", Arguments = "{}" }));
    }
}
