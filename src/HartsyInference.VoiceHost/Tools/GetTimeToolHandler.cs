using System.Globalization;
using System.Text.Json;
using HartsyInference.Tools;

namespace HartsyInference.VoiceHost.Tools;

/// <summary><c>get_time</c>, answered on the host: the current time in UTC and in the host's time zone.</summary>
internal sealed class GetTimeToolHandler(TimeProvider clock) : IToolHandler
{
    public string Name => VoiceHostTools.GetTime;

    public string Description => "Get the current date and time.";

    public string JsonSchema => """{"type":"object","properties":{},"additionalProperties":false}""";

    public Task<string> InvokeAsync(string argumentsJson, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        DateTimeOffset utc = clock.GetUtcNow();
        TimeZoneInfo zone = clock.LocalTimeZone;
        DateTimeOffset local = TimeZoneInfo.ConvertTime(utc, zone);
        ClockReading reading = new()
        {
            Utc = utc.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'", CultureInfo.InvariantCulture),
            Local = local.ToString("yyyy-MM-dd'T'HH:mm:sszzz", CultureInfo.InvariantCulture),
            TimeZone = zone.Id,
            DayOfWeek = local.DayOfWeek.ToString(),
        };
        return Task.FromResult(JsonSerializer.Serialize(reading, ToolJsonContext.Default.ClockReading));
    }
}
