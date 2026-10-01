using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.Engine;
using HartsyInference.Engine.Services;
using HartsyInference.PhoneLink;
using HartsyInference.Tools;
using HartsyInference.Tools.Parsing;
using HartsyInference.Voice;
using HartsyInference.VoiceHost.Calls;
using HartsyInference.VoiceHost.Config;
using HartsyInference.VoiceHost.Link;
using HartsyInference.VoiceHost.Runtime;
using HartsyInference.VoiceHost.Tools;
using Microsoft.Extensions.Hosting;

namespace HartsyInference.VoiceHost;

/// <summary>The host's lifetime as a hosted service: at start the engine on the audio card (tool-call parsing installed,
/// the language model reached through <see cref="VoiceAgentOptions.LlmDevice"/> on every request), the voice models
/// loaded and warmed, then the PhoneLink socket; at stop every call ended with a reason the gateway hears, then the
/// models and the engine released.</summary>
public sealed class VoiceHostService : IHostedService
{
    private readonly VoiceHostSettings _settings;
    private readonly Func<InferenceEngine, ITextService> _text;
    private InferenceEngine? _engine;
    private VoiceModelSet? _models;
    private PhoneLinkServer? _server;

    /// <param name="settings">The loaded configuration.</param>
    /// <param name="text">The language model the sessions answer with; the engine's own text service in production.</param>
    public VoiceHostService(VoiceHostSettings settings, Func<InferenceEngine, ITextService> text)
    {
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentNullException.ThrowIfNull(text);
        _settings = settings;
        _text = text;
    }

    /// <summary>The socket server, once started; for tests.</summary>
    internal PhoneLinkServer? Server => _server;

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        VoiceHostConfig config = _settings.Config;
        VoiceAgentOptions agent = _settings.Agent;
        if (config.Engine.SettingsFile.Length > 0)
        {
            // Before any engine setting is read: the file is loaded once, on the first read.
            KnobFile.ExplicitPath = config.Engine.SettingsFile;
        }
        HostRuntimeTuning.ApplyThreadPool(config.Engine.CpuThreadCap);
        long started = MonotonicClock.NowNs();
        try
        {
            EngineOptions engineOptions = new();
            ToolCalling.Install(engineOptions, ToolCallFormats.Detect(agent.LlmModel));
            _engine = new InferenceEngine(BackendFactory.Kind(agent.AudioDevice), BackendFactory.ParseOrdinal(agent.AudioDevice), engineOptions);
            Logs.Info($"[VoiceHost] Engine on {_engine.ComputeBackend.Device} ({_engine.ComputeBackend.Capabilities.DeviceName}); "
                + $"the language model {agent.LlmModel} runs on {agent.LlmDevice}.");
            _models = await VoiceModelSet.LoadAsync(_engine, agent, config.Models.WakeModelRoot, cancellationToken).ConfigureAwait(false);
            ITextService text = _text(_engine);
            await WarmModelsAsync(_models, text, config.Tools.Enabled, cancellationToken).ConfigureAwait(false);
            Logs.Info($"[VoiceHost] {agent.SttModel}, {agent.TtsModel} and {agent.LlmModel} loaded and warm in "
                + $"{(MonotonicClock.NowNs() - started) / 1_000_000} ms; denoise {(_models.DenoiseEnabled ? "on" : "off")}.");
            _server = new PhoneLinkServer(ServerOptions(_settings), new VoiceAgentCallFactory(_models, text, agent));
            _server.Start();
        }
        catch
        {
            await ReleaseAsync().ConfigureAwait(false);
            throw;
        }
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        Logs.Info("[VoiceHost] Stopping: ending calls, then releasing the models.");
        await ReleaseAsync().ConfigureAwait(false);
    }

    /// <summary>Warms <paramref name="models"/> with the same tool definitions a real call on <paramref name="enabledTools"/>
    /// would offer (<see cref="VoiceHostTools.Build"/>, via <see cref="VoiceHostTools.WarmDefinitions"/>), so the
    /// tool-call grammar sampler, its stream filter/parser and the chat template's tools branch are hot before the
    /// first caller, not just the cold one-token path a tool-less warm-up takes.</summary>
    internal static Task WarmModelsAsync(VoiceModelSet models, ITextService text, IReadOnlyList<string> enabledTools, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(models);
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(enabledTools);
        return models.WarmAsync(text, VoiceHostTools.WarmDefinitions(enabledTools), cancel);
    }

    /// <summary>The server options <paramref name="settings"/> describe.</summary>
    internal static PhoneLinkServerOptions ServerOptions(VoiceHostSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);
        VoiceHostConfig config = settings.Config;
        return new PhoneLinkServerOptions
        {
            SocketPath = config.Link.SocketPath,
            SocketMode = settings.SocketMode,
            Token = settings.LinkToken,
            PrebufferMs = config.Link.PrebufferMs,
            LivenessTimeoutMs = config.Link.LivenessTimeoutSeconds * 1000,
            ToolTimeoutMs = config.Tools.TimeoutMs,
            Tools = config.Tools.Enabled,
            Greeting = string.IsNullOrWhiteSpace(config.Agent.Greeting) ? null : config.Agent.Greeting,
            ResumeApology = string.IsNullOrWhiteSpace(config.Agent.ResumeApology) ? null : config.Agent.ResumeApology,
        };
    }

    private async Task ReleaseAsync()
    {
        PhoneLinkServer? server = Interlocked.Exchange(ref _server, null);
        if (server is not null)
        {
            await server.StopAsync(LinkCallEndReason.LocalHangup).ConfigureAwait(false);
        }
        VoiceModelSet? models = Interlocked.Exchange(ref _models, null);
        if (models is not null)
        {
            await models.DisposeAsync().ConfigureAwait(false);
        }
        Interlocked.Exchange(ref _engine, null)?.Dispose();
    }
}
