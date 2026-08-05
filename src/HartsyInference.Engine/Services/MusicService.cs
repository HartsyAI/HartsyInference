using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Text-to-music service: picks a descriptor from the model spec and runs the generation on the shared audio
/// device under the generation lock. Covers MusicGen, AudioGen, ACE-Step, YuE, and HeartMuLa.</summary>
public sealed class MusicService : IMusicService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal MusicService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<AudioResult> GenerateAsync(ModelSpec spec, MusicRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (string.IsNullOrWhiteSpace(request.Prompt) && string.IsNullOrWhiteSpace(request.Genre))
        {
            // ACE-Step puts the style in genre and the (optional) lyrics in prompt, so either alone is enough.
            throw new ArgumentException("No prompt or genre supplied to generate music.", nameof(request));
        }
        AudioModelSelector selector = AudioModelSelector.Parse(spec);
        ValidateEditingModes(request, selector.Id);
        MusicModelDescriptor descriptor = MusicCatalog.Resolve(selector.Id);
        IBackend backend = _engine.Backend;
        MusicLoadContext loadContext = BuildLoadContext(backend);
        string key = descriptor.CacheKey(selector) + loadContext.CacheSuffix();

        return _engine.AudioRuntime.RunAsync(backend, $"music:{key}", async ct =>
        {
            IMusicRunner runner = await _engine.AudioRuntime.Music
                .GetOrLoadAsync(key, token => descriptor.LoadAsync(loadContext, selector, token), ct).ConfigureAwait(false);
            ct.ThrowIfCancellationRequested();
            long started = Environment.TickCount64;
            MusicAudio audio = runner.Synthesize(backend, request, ct);
            if (audio.Left is null || audio.Left.Length == 0)
            {
                throw new InvalidOperationException("The music model produced no audio.");
            }
            double seconds = AudioClipCodec.Seconds(audio.Left.Length, runner.SampleRate);
            Logs.Verbose($"[Audio][Music] Generated {seconds:0.0}s @ {runner.SampleRate} Hz "
                + $"({(audio.Right is null ? "mono" : "stereo")}) in {Environment.TickCount64 - started}ms.");
            return new AudioResult
            {
                Data = AudioClipCodec.EncodeWav(audio.Left, audio.Right, runner.SampleRate),
                Format = "wav",
                DurationSeconds = seconds,
                SampleRate = runner.SampleRate,
                Meta = new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["model"] = key,
                    ["seed"] = request.Seed.ToString(CultureInfo.InvariantCulture),
                    ["channels"] = audio.Right is null ? "1" : "2",
                },
            };
        }, cancel);
    }

    /// <summary>Builds the load-time context: single-device Q4_K (byte-identical to pre-placement behavior)
    /// unless the engine placement has ≥2 <c>ShardDevices</c>, in which case the big-LM loaders (YuE) get the
    /// resolved shard backends and default to un-quantized weights pooled across them.</summary>
    private MusicLoadContext BuildLoadContext(IBackend primary)
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
        return new MusicLoadContext
        {
            Backend = primary,
            ShardStages = stages,
            ShardRatios = sharded ? _engine.Placement.ShardRatios : null,
            LmQuant = AudioLmQuantPolicy.Resolve(sharded),
        };
    }

    /// <summary>Gates the audio-conditioned editing modes: they are mutually exclusive, and only ACE-Step 1.5 has an
    /// edit path (its DiT reads <c>context_latents = [src_latent ‖ chunk_mask]</c>) — every other music family is
    /// text-conditioned only and is refused by name. ACE-Step requests fall through to
    /// <c>AceStepMusicModel</c>, which decodes the clip and builds the src/mask/start-sigma plan.
    /// <b>Parity-pending:</b> the edit modes have not been validated against real weights, and cover approximates
    /// upstream's FSQ-detokenized 5 Hz hints with raw 25 Hz Oobleck latents.</summary>
    private static void ValidateEditingModes(MusicRequest request, string modelId)
    {
        int selected = (request.Continuation is not null ? 1 : 0) + (request.Repaint is not null ? 1 : 0)
            + (request.Cover is not null ? 1 : 0);
        if (selected == 0)
        {
            return;
        }
        string mode = request.Continuation is not null ? "Continuation"
            : request.Repaint is not null ? "Repaint"
            : "Cover";
        if (selected > 1)
        {
            string set = string.Join(", ",
                new[]
                {
                    request.Continuation is not null ? "Continuation" : null,
                    request.Repaint is not null ? "Repaint" : null,
                    request.Cover is not null ? "Cover" : null,
                }.Where(name => name is not null));
            throw new ArgumentException(
                $"Music editing modes are mutually exclusive, but {set} were all supplied — set exactly one.", nameof(request));
        }
        if (!string.Equals(modelId, AudioWeightsCatalog.AceStepId, StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException(
                $"Music '{mode.ToLowerInvariant()}' is an ACE-Step-only editing mode; the '{modelId}' family "
                + "(MusicGen / AudioGen / YuE / HeartMuLa) has no audio-conditioned edit path in this engine. "
                + "Drop the Continuation/Repaint/Cover input or select 'acestep'.");
        }
    }
}
