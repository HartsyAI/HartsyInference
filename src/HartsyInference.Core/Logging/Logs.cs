namespace HartsyInference.Core.Logging;

/// <summary>Static, low-overhead logging for HartsyInference; redirectable to any framework via <see cref="SetLogger"/>.</summary>
public static class Logs
{
    private static Action<LogLevel, string>? _logger;

    /// <summary>Minimum log level to emit. Default is <see cref="LogLevel.Info"/>.</summary>
    public static LogLevel MinLevel { get; set; } = LogLevel.Info;

    /// <summary>Sets a custom logger callback. If not set, logs are written to <see cref="Console.Error"/>.</summary>
    public static void SetLogger(Action<LogLevel, string> logger)
    {
        _logger = logger;
    }

    /// <summary>Logs at <see cref="LogLevel.Verbose"/> when permitted by <see cref="MinLevel"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Verbose(string message)
    {
        if (MinLevel <= LogLevel.Verbose) Write(LogLevel.Verbose, message);
    }

    /// <summary>Logs at <see cref="LogLevel.Debug"/> when permitted by <see cref="MinLevel"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Debug(string message)
    {
        if (MinLevel <= LogLevel.Debug) Write(LogLevel.Debug, message);
    }

    /// <summary>Logs at <see cref="LogLevel.Info"/> when permitted by <see cref="MinLevel"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Info(string message)
    {
        if (MinLevel <= LogLevel.Info) Write(LogLevel.Info, message);
    }

    /// <summary>Logs at <see cref="LogLevel.Warning"/> when permitted by <see cref="MinLevel"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Warning(string message)
    {
        if (MinLevel <= LogLevel.Warning) Write(LogLevel.Warning, message);
    }

    /// <summary>Logs at <see cref="LogLevel.Error"/> when permitted by <see cref="MinLevel"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Error(string message)
    {
        if (MinLevel <= LogLevel.Error) Write(LogLevel.Error, message);
    }

    /// <summary>Logs at <see cref="LogLevel.Error"/> with the exception appended, when permitted by <see cref="MinLevel"/>.</summary>
    public static void Error(string message, Exception ex)
    {
        if (MinLevel <= LogLevel.Error) Write(LogLevel.Error, $"{message}: {ex}");
    }

    private static void Write(LogLevel level, string message)
    {
        if (_logger is not null)
        {
            _logger(level, message);
            return;
        }

        string prefix = level switch
        {
            LogLevel.Verbose => "[VRB]",
            LogLevel.Debug => "[DBG]",
            LogLevel.Info => "[INF]",
            LogLevel.Warning => "[WRN]",
            LogLevel.Error => "[ERR]",
            _ => "[???]",
        };

        Console.Error.WriteLine($"{prefix} {message}");
    }
}
