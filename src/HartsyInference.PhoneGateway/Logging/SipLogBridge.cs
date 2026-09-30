using HartsyInference.Core.Logging;
using Microsoft.Extensions.Logging;
using MelLogLevel = Microsoft.Extensions.Logging.LogLevel;

namespace HartsyInference.PhoneGateway.Logging;

/// <summary>Routes sipsorcery's <c>Microsoft.Extensions.Logging</c> output into <see cref="Logs"/>. Trace and Debug
/// are dropped unless <c>logging.sipDebug</c> is on: sipsorcery's transport trace prints whole SIP messages,
/// Authorization headers included, so it is never on by default.</summary>
internal sealed class SipLogBridge(bool debug) : ILoggerFactory
{
    /// <summary>Installs the bridge as sipsorcery's logger factory.</summary>
    public static void Install(bool debug) => SIPSorcery.LogFactory.Set(new SipLogBridge(debug));

    public ILogger CreateLogger(string categoryName) => new SipLogger(categoryName, debug);

    public void AddProvider(ILoggerProvider provider)
    {
    }

    public void Dispose()
    {
    }

    private sealed class SipLogger(string category, bool debug) : ILogger
    {
        private readonly string _prefix = "[sip:" + ShortCategory(category) + "] ";

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(MelLogLevel level) => level switch
        {
            MelLogLevel.Trace or MelLogLevel.Debug => debug && Logs.MinLevel <= Core.Logging.LogLevel.Debug,
            MelLogLevel.Information => Logs.MinLevel <= Core.Logging.LogLevel.Info,
            MelLogLevel.Warning => Logs.MinLevel <= Core.Logging.LogLevel.Warning,
            MelLogLevel.Error or MelLogLevel.Critical => Logs.MinLevel <= Core.Logging.LogLevel.Error,
            _ => false,
        };

        public void Log<TState>(MelLogLevel level, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            if (!IsEnabled(level))
            {
                return;
            }
            string message = _prefix + formatter(state, exception);
            switch (level)
            {
                case MelLogLevel.Trace:
                case MelLogLevel.Debug:
                    Logs.Debug(message);
                    break;
                case MelLogLevel.Information:
                    Logs.Info(message);
                    break;
                case MelLogLevel.Warning:
                    Logs.Warning(message);
                    break;
                default:
                    if (exception is null)
                    {
                        Logs.Error(message);
                    }
                    else
                    {
                        Logs.Error(message, exception);
                    }
                    break;
            }
        }

        private static string ShortCategory(string category)
        {
            int dot = category.LastIndexOf('.');
            return dot < 0 ? category : category.Substring(dot + 1);
        }
    }
}
