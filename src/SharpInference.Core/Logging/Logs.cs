namespace SharpInference.Core.Logging;

/// <summary>Static logging class for SharpInference. Provides simple, low-overhead logging that can be redirected to any logging framework via <see cref="SetLogger"/>.</summary>
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Verbose(string message)
    {
        if (MinLevel <= LogLevel.Verbose) Write(LogLevel.Verbose, message);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Debug(string message)
    {
        if (MinLevel <= LogLevel.Debug) Write(LogLevel.Debug, message);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Info(string message)
    {
        if (MinLevel <= LogLevel.Info) Write(LogLevel.Info, message);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Warning(string message)
    {
        if (MinLevel <= LogLevel.Warning) Write(LogLevel.Warning, message);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Error(string message)
    {
        if (MinLevel <= LogLevel.Error) Write(LogLevel.Error, message);
    }

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
