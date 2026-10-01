using HartsyInference.Core.Logging;
using HartsyInference.Engine;
using HartsyInference.Engine.Services;
using HartsyInference.VoiceHost.Config;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace HartsyInference.VoiceHost;

/// <summary>Runs the voice host as a generic host: <see cref="VoiceHostService"/> as its only hosted service, SIGTERM and
/// Ctrl+C handled by the console lifetime with a bounded graceful stop. Built empty, so no configuration source reads the
/// environment or an appsettings file: everything comes from <c>voice.json</c>.</summary>
public static class VoiceHostProgram
{
    /// <summary>How long a stop may take: ending calls, then releasing the models. The unit's <c>TimeoutStopSec</c> is
    /// longer.</summary>
    public static readonly TimeSpan ShutdownTimeout = TimeSpan.FromSeconds(20);

    /// <summary>Runs until stopped. Returns 0 after a clean stop and 1 when start-up or the host failed.</summary>
    /// <param name="settings">The loaded configuration.</param>
    /// <param name="text">The language model every session answers with; <c>engine =&gt; engine.Text</c> in production.</param>
    /// <param name="cancel">Stops the host, as SIGTERM does.</param>
    public static async Task<int> RunAsync(VoiceHostSettings settings, Func<InferenceEngine, ITextService> text, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentNullException.ThrowIfNull(text);
        using IHost host = Build(settings, text);
        try
        {
            await host.RunAsync(cancel).ConfigureAwait(false);
            return 0;
        }
        catch (OperationCanceledException) when (cancel.IsCancellationRequested)
        {
            return 0;
        }
        catch (Exception ex)
        {
            // The process boundary: report why the host could not run, and let the supervisor restart it.
            Logs.Error("[VoiceHost] The voice host stopped on a failure", ex);
            return 1;
        }
    }

    /// <summary>The generic host for <paramref name="settings"/>, not started.</summary>
    internal static IHost Build(VoiceHostSettings settings, Func<InferenceEngine, ITextService> text)
    {
        HostApplicationBuilder builder = Host.CreateEmptyApplicationBuilder(new HostApplicationBuilderSettings { ApplicationName = "HartsyInference.VoiceHost" });
        builder.Services.Configure<HostOptions>(options => options.ShutdownTimeout = ShutdownTimeout);
        builder.Services.AddSingleton(settings);
        builder.Services.AddHostedService(_ => new VoiceHostService(settings, text));
        return builder.Build();
    }
}
