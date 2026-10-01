namespace HartsyInference.VoiceHost.Tools;

/// <summary>The <c>get_time</c> result: ISO-8601 times in UTC and in the host's zone, the zone id and the local weekday.</summary>
internal sealed record ClockReading
{
    public required string Utc { get; init; }

    public required string Local { get; init; }

    public required string TimeZone { get; init; }

    public required string DayOfWeek { get; init; }
}
