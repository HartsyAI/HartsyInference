using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.CompilerServices;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Text-to-speech service: picks a descriptor from the model spec, materializes the optional voice reference, and runs the synthesis on the shared audio device under the generation lock. A lease hands the same resident runner to a caller that gates the device itself.</summary>
public sealed class SpeechService : ISpeechService
{
    /// <summary>Voice references are decoded at 24 kHz — the rate the cloning models expect.</summary>
    private const int ReferenceSampleRate = 24_000;

    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal SpeechService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<AudioResult> SynthesizeAsync(ModelSpec spec, SpeechRequest request, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (string.IsNullOrWhiteSpace(request.Text))
        {
            throw new ArgumentException("No text supplied to synthesize.", nameof(request));
        }
        TtsTarget target = ResolveTarget(spec, request.Voice);
        return _engine.AudioRuntime.RunAsync(target.Backend, Job(target), async ct =>
        {
            (float[]? referenceMono, string? referenceWavPath) = MaterializeReference(request.Reference);
            try
            {
                ct.ThrowIfCancellationRequested();
                ITtsRunner runner = await _engine.AudioRuntime.Tts
                    .GetOrLoadAsync(target.Key, token => target.Descriptor.LoadAsync(target.LoadContext, target.Variant, token), ct)
                    .ConfigureAwait(false);
                TtsJob job = BuildJob(request.Text, request, referenceMono, referenceWavPath);
                long started = Environment.TickCount64;
                float[] samples = runner.Synthesize(target.Backend, job);
                if (samples is null || samples.Length == 0)
                {
                    throw new InvalidOperationException("The text-to-speech model produced no audio.");
                }
                double seconds = AudioClipCodec.Seconds(samples.Length, runner.SampleRate);
                Logs.Verbose($"[Audio][TTS] Synthesized {seconds:0.0}s @ {runner.SampleRate} Hz in {Environment.TickCount64 - started}ms.");
                return new AudioResult
                {
                    Data = AudioClipCodec.EncodeWav(samples, null, runner.SampleRate),
                    Format = "wav",
                    DurationSeconds = seconds,
                    SampleRate = runner.SampleRate,
                    Meta = new Dictionary<string, string>(StringComparer.Ordinal)
                    {
                        ["model"] = target.Key,
                        ["seed"] = request.Seed.ToString(CultureInfo.InvariantCulture),
                    },
                };
            }
            finally
            {
                DeleteTempReference(referenceWavPath);
            }
        }, cancel, stageBackends: target.StageBackends);
    }

    /// <inheritdoc/>
    public IAsyncEnumerable<AudioChunk> SynthesizeStreamAsync(ModelSpec spec, SpeechRequest request, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (string.IsNullOrWhiteSpace(request.Text))
        {
            throw new ArgumentException("No text supplied to synthesize.", nameof(request));
        }
        return StreamCore(ResolveTarget(spec, request.Voice), request, cancel);
    }

    /// <inheritdoc/>
    public async Task<ISynthesizerLease> OpenSynthesizerAsync(ModelSpec spec, CancellationToken cancel = default)
    {
        TtsTarget target = ResolveTarget(spec, voice: null);
        AudioRuntime runtime = _engine.AudioRuntime;
        string? weightsVoice = target.Descriptor.VoiceSelectsWeights ? target.Variant : null;
        return await runtime.OpenLeaseAsync(target.Backend, runtime.Tts, target.Key,
            token => target.Descriptor.LoadAsync(target.LoadContext, target.Variant, token),
            runner => new SynthesizerLease(runtime, target.Key, runner, target.Backend, weightsVoice),
            cancel, target.StageBackends, target.EstimateWeightBytes).ConfigureAwait(false);
    }

    /// <summary>Resolves the runner a request names: descriptor, load variant, backend, load context and cache key. The
    /// one formula the service calls and the leases share, so a lease and a service call on one spec meet in one cache
    /// entry.</summary>
    private TtsTarget ResolveTarget(ModelSpec spec, string? voice)
    {
        AudioModelSelector selector = AudioModelSelector.Parse(spec);
        TtsModelDescriptor descriptor = TtsCatalog.Resolve(selector.Id);
        string variant = ResolveVariant(selector, descriptor, voice);
        string repo = descriptor.ResolveRepo(variant);
        IBackend backend = _engine.Backend;
        TtsLoadContext loadContext = BuildLoadContext(backend);
        string key = repo + (descriptor.VoiceSelectsWeights ? "|" + variant : "") + loadContext.CacheSuffix();
        IReadOnlyList<IBackend>? stageBackends = loadContext.ShardStages is { Count: >= 2 } stages ? [.. stages.Select(s => s.Backend)] : null;
        // A layer-split load spreads its weights over several devices, so the whole checkpoint is no single device's need.
        Func<long>? estimate = loadContext.IsSharded ? null
            : () => AudioWeightFootprint.Estimate(repo, "tts", descriptor.PromotesHalfToF32);
        return new TtsTarget(descriptor, variant, backend, loadContext, key, stageBackends, estimate);
    }

    /// <summary>The runtime job for <paramref name="target"/>'s runner.</summary>
    private AudioJob Job(TtsTarget target) => new(_engine.AudioRuntime.Tts, target.Key, target.EstimateWeightBytes);

    /// <summary>Resolves the load variant for a <see cref="TtsModelDescriptor"/>: the real sub-variant the
    /// caller named -- either via <paramref name="voice"/>, or via <c>":variant"</c> in the request token,
    /// already captured in <paramref name="selector"/> -- or, for a <see cref="TtsModelDescriptor.VoiceSelectsWeights"/>
    /// descriptor with neither, the empty string signaling "no real voice; apply your own default."
    ///
    /// <para><see cref="AudioModelSelector.Parse"/> falls back to the bare catalog token (e.g. <c>"piper"</c>)
    /// for <see cref="AudioModelSelector.Variant"/> whenever the request token has no <c>':'</c> -- correct
    /// for a descriptor that treats a bare id as its own repo/model identifier (see that type's doc), but for
    /// a <c>VoiceSelectsWeights</c> descriptor that fallback is NOT a voice: Piper's catalog id is
    /// <c>"piper"</c>, and <c>"piper"</c> is not a file in <c>rhasspy/piper-voices</c>. An unparameterized
    /// request used to resolve <paramref name="selector"/>'s bare-token <c>Variant</c> straight through as if
    /// it named a real voice, and the Engine 404'd fetching <c>piper.onnx</c> instead of falling back to
    /// Piper's own documented default voice.
    ///
    /// <para>Detected by comparing <see cref="AudioModelSelector.Variant"/> to <see cref="AudioModelSelector.Id"/>
    /// rather than against the literal string <c>"piper"</c>, so the same fix covers any other
    /// <c>VoiceSelectsWeights</c> model with this shape, not just Piper. The empty-string result (not null)
    /// keeps every existing consumer of this variant -- the cache <c>key</c> above, each descriptor's own
    /// <c>LoadAsync</c> (already treating empty/<c>"default"</c> as "use my default" where that applies, e.g.
    /// Piper's), and <see cref="SynthesizerLease"/>'s voice-mismatch check -- working exactly as they did,
    /// with a type that is never actually a voice instead of one that merely looks like it could be.</para>
    ///
    /// <para><c>internal</c> so this exact decision -- not the whole <see cref="ResolveTarget"/>, which needs
    /// a live <see cref="InferenceEngine"/> for its backend/placement lookups -- has its own fast, pure unit
    /// tests.</para></summary>
    internal static string ResolveVariant(AudioModelSelector selector, TtsModelDescriptor descriptor, string? voice)
    {
        if (!descriptor.VoiceSelectsWeights)
        {
            return selector.Variant;
        }
        if (IsNamedVoice(voice))
        {
            return voice;
        }
        bool bareTokenFallback = string.Equals(selector.Variant, selector.Id, StringComparison.OrdinalIgnoreCase);
        return bareTokenFallback ? "" : selector.Variant;
    }

    /// <summary>Whether <paramref name="voice"/> names a voice rather than the model default; "default" is a placeholder
    /// some callers send.</summary>
    internal static bool IsNamedVoice([NotNullWhen(true)] string? voice) =>
        !string.IsNullOrWhiteSpace(voice) && !voice.Equals("default", StringComparison.OrdinalIgnoreCase);

    /// <summary>Owns reference-file materialization/cleanup around the streamed run, exactly mirroring <see cref="SynthesizeAsync"/>'s non-streaming <c>finally</c>.</summary>
    private async IAsyncEnumerable<AudioChunk> StreamCore(TtsTarget target, SpeechRequest request,
        [EnumeratorCancellation] CancellationToken cancel)
    {
        (float[]? referenceMono, string? referenceWavPath) = MaterializeReference(request.Reference);
        try
        {
            TtsJob job = BuildJob(request.Text, request, referenceMono, referenceWavPath);
            await foreach (AudioChunk chunk in _engine.AudioRuntime.RunStreamAsync(target.Backend, Job(target),
                ct => StreamWork(target, job, ct), cancel, target.StageBackends).ConfigureAwait(false))
            {
                yield return chunk;
            }
        }
        finally
        {
            DeleteTempReference(referenceWavPath);
        }
    }

    /// <summary>Loads the runner and streams from it if it implements <see cref="IStreamingTtsRunner"/>; every other model falls back to one chunk containing the complete synthesized buffer, so <see cref="StreamCore"/> and its caller have a single code path regardless of which model is selected.</summary>
    private async IAsyncEnumerable<AudioChunk> StreamWork(TtsTarget target, TtsJob job, [EnumeratorCancellation] CancellationToken ct)
    {
        IBackend backend = target.Backend;
        string key = target.Key;
        ITtsRunner runner = await _engine.AudioRuntime.Tts
            .GetOrLoadAsync(key, token => target.Descriptor.LoadAsync(target.LoadContext, target.Variant, token), ct)
            .ConfigureAwait(false);
        long started = Environment.TickCount64;
        if (runner is IStreamingTtsRunner streaming)
        {
            await foreach (AudioChunk chunk in streaming.SynthesizeStream(backend, job, ct).ConfigureAwait(false))
            {
                yield return chunk;
            }
            Logs.Verbose($"[Audio][TTS] Streamed '{key}' in {Environment.TickCount64 - started}ms.");
        }
        else
        {
            float[] samples = runner.Synthesize(backend, job);
            if (samples is null || samples.Length == 0)
            {
                throw new InvalidOperationException("The text-to-speech model produced no audio.");
            }
            double seconds = AudioClipCodec.Seconds(samples.Length, runner.SampleRate);
            Logs.Verbose($"[Audio][TTS] Synthesized {seconds:0.0}s @ {runner.SampleRate} Hz in {Environment.TickCount64 - started}ms (single-chunk fallback, no streaming support).");
            yield return new AudioChunk(samples, runner.SampleRate, Channels: 1, StartSampleOffset: 0);
        }
    }

    /// <summary>Decodes a voice reference to mono 24 kHz samples plus a temp WAV for pipelines that take a file path. Returns <c>(null, null)</c> when no reference was supplied. Shared by the batch and streaming synthesis paths and the synthesizer lease.</summary>
    internal static (float[]? Mono, string? WavPath) MaterializeReference(AudioClip? reference)
    {
        if (reference is null || reference.Data.Length == 0)
        {
            return (null, null);
        }
        float[] mono = AudioClipCodec.DecodeMono(reference, ReferenceSampleRate);
        string wavPath = Path.Combine(Path.GetTempPath(), $"hartsy_voiceref_{Guid.NewGuid():N}.wav");
        using (FileStream file = new FileStream(wavPath, FileMode.Create, FileAccess.Write))
        {
            WavFile.WriteMono16(file, mono, ReferenceSampleRate);
        }
        return (mono, wavPath);
    }

    /// <summary>Builds the per-model job for <paramref name="text"/> from the request's knobs plus the already-materialized reference. Shared by the batch and streaming synthesis paths and the synthesizer lease so their knob-mapping never drifts apart.</summary>
    internal static TtsJob BuildJob(string text, SpeechRequest request, float[]? referenceMono, string? referenceWavPath) => new TtsJob
    {
        Text = text,
        RefText = request.RefText,
        Reference = request.Reference,
        ReferenceMono24k = referenceMono,
        ReferenceWavPath = referenceWavPath,
        // "default" is a placeholder some callers send, not a real voice name.
        Voice = string.IsNullOrEmpty(request.Voice) || request.Voice.Equals("default", StringComparison.OrdinalIgnoreCase)
            ? null : request.Voice,
        Speed = request.Speed,
        Exaggeration = request.Exaggeration,
        NfeStep = request.NfeStep,
        CfgScale = request.CfgScale,
        SpeakerId = request.SpeakerId,
        NormalizeLoudness = request.NormalizeLoudness,
        Temperature = request.Temperature,
        WaveformTemperature = request.WaveformTemperature,
        TopP = request.TopP,
        TopK = request.TopK,
        MaxTokens = request.MaxTokens,
        Emotion = request.Emotion,
        SpeakingRate = request.SpeakingRate,
        PitchStd = request.PitchStd,
        Seed = request.Seed,
    };

    /// <summary>Builds the load-time context: single-device (byte-identical to pre-placement behavior) unless the engine placement has ≥2 <c>ShardDevices</c>, in which case CosyVoice's Qwen2 LM gets the resolved shard backends and layer-splits across them; every other TTS family ignores the shard stages. Mirrors <c>MusicService.BuildLoadContext</c>.</summary>
    private TtsLoadContext BuildLoadContext(IBackend primary)
    {
        IReadOnlyList<string> shardDevices = _engine.Placement.ShardDevices;
        bool sharded = shardDevices.Count >= 2;
        List<(string Selector, IBackend Backend)>? stages = null;
        if (sharded)
        {
            stages = new List<(string, IBackend)>(shardDevices.Count);
            foreach (string device in shardDevices)
            {
                stages.Add((device, _engine.EnsureBackend(device)));
            }
        }
        return new TtsLoadContext
        {
            Backend = primary,
            ShardStages = stages,
            ShardRatios = sharded ? _engine.Placement.ShardRatios : null,
        };
    }

    /// <summary>Deletes the temp WAV <see cref="MaterializeReference"/> wrote, logging rather than throwing on failure.</summary>
    internal static void DeleteTempReference(string? path)
    {
        if (path is null)
        {
            return;
        }
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (Exception ex)
        {
            Logs.Warning($"[Audio][TTS] Failed to delete the temp voice reference '{path}': {ex.Message}");
        }
    }

    /// <summary>What <see cref="ResolveTarget"/> resolved: the runner's descriptor, load variant, backend, load context,
    /// cache key, for a layer-split load the stage backends to gate, and the weight sizing for the switch check.</summary>
    private readonly record struct TtsTarget(TtsModelDescriptor Descriptor, string Variant, IBackend Backend,
        TtsLoadContext LoadContext, string Key, IReadOnlyList<IBackend>? StageBackends, Func<long>? EstimateWeightBytes);
}
