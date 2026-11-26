using System.Globalization;
using HartsyInference.Audio.Models.Separation;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Audio-effects service: Demucs stem separation from a placed checkpoint, and Resemble-Enhance speech enhancement, both run on the shared audio device under the generation lock.</summary>
public sealed class FxService : IFxService
{
    private readonly InferenceEngine _engine;

    // Demucs (DemucsSpec.cs) calls IBackend.Stft directly, which GPU backends do not implement (CudaBackend
    // throws "CUDA STFT not supported - use CPU backend for audio") — the CLI's fx commands hard-force CPU
    // for exactly this reason, and server/SwarmUI requests must too or every non-CLI Demucs run on a CUDA
    // host fails at first STFT. Resemble-Enhance does NOT share this constraint (see EnhanceAsync).
    private static readonly Lazy<IBackend> _cpuBackend = new(() => BackendFactory.Create("cpu"));

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal FxService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<StemsResult> SeparateAsync(ModelSpec spec, FxSeparateRequest request, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        AudioModelSelector selector = AudioModelSelector.Parse(spec);
        if (IsMdxSeparator(selector, request))
        {
            return SeparateVocalsAsync(selector, request, cancel);
        }
        if (IsVoiceIsolator(selector, request, out bool realtime))
        {
            return IsolateVoiceAsync(selector, request, realtime, cancel);
        }
        string modelName = string.IsNullOrWhiteSpace(request.Model) ? selector.Variant : request.Model;
        // The runner is cached by what identifies its weights: an explicit local checkpoint, else the model name
        // (EnsureDemucsPathAsync resolves the same path for the same pair). The job carries the same key so the
        // memory-pressure sweep can keep this runner.
        string cacheKey = selector.LocalPath ?? modelName;
        IBackend backend = _cpuBackend.Value;

        return _engine.AudioRuntime.RunAsync(backend, new AudioJob(_engine.AudioRuntime.Demucs, cacheKey), async ct =>
        {
            (float[] left, float[] right) = AudioClipCodec.DecodeStereo(request.Audio, FxCatalog.DemucsSampleRate);
            if (left.Length == 0)
            {
                throw new ArgumentException("The audio to separate decoded to no samples.", nameof(request));
            }
            ct.ThrowIfCancellationRequested();

            string path = await FxCatalog.EnsureDemucsPathAsync(modelName, selector.LocalPath, ct).ConfigureAwait(false);
            DemucsRunner runner = await _engine.AudioRuntime.Demucs
                .GetOrLoadAsync(cacheKey, _ => Task.FromResult(FxCatalog.LoadDemucs(path, modelName)), ct).ConfigureAwait(false);
            long started = Environment.TickCount64;
            (float[] Left, float[] Right)[] stems = runner.Separate(backend, left, right,
                request.Shifts, request.Overlap, request.Segment, request.Seed);
            IReadOnlyList<string> names = runner.Sources;
            Dictionary<string, byte[]> encoded = new Dictionary<string, byte[]>(stems.Length, StringComparer.Ordinal);
            for (int i = 0; i < stems.Length; i++)
            {
                string name = i < names.Count ? names[i] : $"stem{i}";
                encoded[name] = AudioClipCodec.EncodeWav(stems[i].Left, stems[i].Right, runner.SampleRate);
            }
            Logs.Verbose($"[Audio][Demucs] Separated {AudioClipCodec.Seconds(left.Length, FxCatalog.DemucsSampleRate):0.0}s "
                + $"into {encoded.Count} stems in {Environment.TickCount64 - started}ms.");
            return new StemsResult
            {
                Stems = encoded,
                Format = "wav",
                SampleRate = runner.SampleRate,
            };
        }, cancel);
    }

    private static bool IsVoiceIsolator(AudioModelSelector selector, FxSeparateRequest request, out bool realtime)
    {
        string id = string.IsNullOrWhiteSpace(request.Model) ? selector.Id : request.Model;
        realtime = string.Equals(id, FxCatalog.VoiceIsolatorRealtimeId, StringComparison.OrdinalIgnoreCase);
        return realtime || string.Equals(id, FxCatalog.VoiceIsolatorId, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Voice-isolator route of <see cref="SeparateAsync"/>: a <c>voice</c> and a <c>background</c> stem (mono, 48 kHz; the
    /// background is the input minus the voice). Always runs on the CPU backend with the STFT on the host.</summary>
    private Task<StemsResult> IsolateVoiceAsync(AudioModelSelector selector, FxSeparateRequest request, bool realtime, CancellationToken cancel)
    {
        if (request.Shifts != 0 || request.Segment is not null || request.Overlap is not null)
        {
            throw new ArgumentException("Shifts, Segment and Overlap are Demucs options; the voice isolator takes only LookaheadFrames (realtime).", nameof(request));
        }
        if (request.LookaheadFrames is not null && !realtime)
        {
            throw new ArgumentException("LookaheadFrames applies to voice-isolator-realtime only.", nameof(request));
        }
        if (request.LookaheadFrames is { } trained && Array.IndexOf(FxCatalog.VoiceIsolatorRealtimeLookaheads, trained) < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), trained,
                $"LookaheadFrames must be one of {string.Join(", ", FxCatalog.VoiceIsolatorRealtimeLookaheads)}.");
        }
        string cacheKey = selector.LocalPath ?? (realtime ? FxCatalog.VoiceIsolatorRealtimeId : FxCatalog.VoiceIsolatorId);
        // CPU for now: the STFT, the per-band layers and the scan reference run on the host, and the GPU route has not been run.
        IBackend backend = _cpuBackend.Value;
        return _engine.AudioRuntime.RunAsync(backend, new AudioJob(_engine.AudioRuntime.Isolator, cacheKey), async ct =>
        {
            float[] mono = AudioClipCodec.DecodeMono(request.Audio, FxCatalog.VoiceIsolatorSampleRate);
            if (mono.Length == 0)
            {
                throw new ArgumentException("The audio to isolate decoded to no samples.", nameof(request));
            }
            ct.ThrowIfCancellationRequested();
            string path = await FxCatalog.EnsureVoiceIsolatorPathAsync(realtime, selector.LocalPath, ct).ConfigureAwait(false);
            VoiceIsolatorRunner runner = await _engine.AudioRuntime.Isolator
                .GetOrLoadAsync(cacheKey, _ => Task.FromResult(FxCatalog.LoadVoiceIsolator(path, realtime)), ct).ConfigureAwait(false);
            long started = Environment.TickCount64;
            (float[] voice, float[] background) = runner.Isolate(backend, mono, request.LookaheadFrames, LogIsolatorProgress, ct);
            Logs.Verbose($"[Audio][Voice-Isolator] Isolated {AudioClipCodec.Seconds(mono.Length, FxCatalog.VoiceIsolatorSampleRate):0.0}s "
                + $"({(realtime ? "realtime stream" : "offline")}) in {Environment.TickCount64 - started}ms.");
            return new StemsResult
            {
                Stems = new Dictionary<string, byte[]>(StringComparer.Ordinal)
                {
                    ["voice"] = AudioClipCodec.EncodeWav(voice, null, runner.SampleRate),
                    ["background"] = AudioClipCodec.EncodeWav(background, null, runner.SampleRate),
                },
                Format = "wav",
                SampleRate = runner.SampleRate,
            };
        }, cancel);
    }

    private static bool IsMdxSeparator(AudioModelSelector selector, FxSeparateRequest request)
    {
        string id = string.IsNullOrWhiteSpace(request.Model) ? selector.Id : request.Model;
        return ModelCatalog.Find(id)?.Features.HasFlag(CatalogFeatures.MdxNetSeparator) == true;
    }

    /// <summary>Kim_Vocal_2 route of <see cref="SeparateAsync"/>: one <c>vocals</c> stem (44.1 kHz stereo, peak-normalized to 0.9 as the reference does). The network runs on
    /// the engine's backend when that is CUDA or Vulkan, and falls back to the CPU when the card has no room for it (a resident image or video model).</summary>
    private Task<StemsResult> SeparateVocalsAsync(AudioModelSelector selector, FxSeparateRequest request, CancellationToken cancel)
    {
        RejectDemucsOnlyKnobs(request);
        return WithCpuFallbackAsync(_engine.Backend, _cpuBackend.Value, backend => SeparateVocalsOnAsync(backend, selector, request, cancel), cancel);
    }

    /// <summary>Runs <paramref name="run"/> on <paramref name="engine"/> when that is CUDA or Vulkan, and on <paramref name="cpu"/> otherwise. A GPU run that still ends
    /// out of video memory after the audio runtime's own retry, wrapped or not, runs again on the CPU.</summary>
    internal static async Task<T> WithCpuFallbackAsync<T>(IBackend engine, IBackend cpu, Func<IBackend, Task<T>> run, CancellationToken cancel)
    {
        if (engine.Device.Type is not (DeviceType.Cuda or DeviceType.Vulkan))
        {
            return await run(cpu).ConfigureAwait(false);
        }
        try
        {
            return await run(engine).ConfigureAwait(false);
        }
        catch (Exception ex) when (AudioRuntime.FindOutOfVram(ex) is not null && !cancel.IsCancellationRequested)
        {
            Logs.Warning($"[Audio][Kim-Vocal-2] Not enough video memory on {engine.Device}; separating on the CPU instead, which takes far longer.");
            return await run(cpu).ConfigureAwait(false);
        }
    }

    private Task<StemsResult> SeparateVocalsOnAsync(IBackend backend, AudioModelSelector selector, FxSeparateRequest request, CancellationToken cancel)
    {
        string cacheKey = selector.LocalPath ?? FxCatalog.KimVocal2Id;
        return _engine.AudioRuntime.RunAsync(backend, new AudioJob(_engine.AudioRuntime.Mdx, cacheKey), async ct =>
        {
            (float[] left, float[] right) = AudioClipCodec.DecodeStereoSoxrHq(request.Audio, MdxSettings.KimVocal2.SampleRate);
            if (left.Length == 0)
            {
                throw new ArgumentException("The audio to separate decoded to no samples.", nameof(request));
            }
            ct.ThrowIfCancellationRequested();
            MdxVocalRunner runner = await LoadKimVocal2Async(selector, cacheKey, ct).ConfigureAwait(false);
            long started = Environment.TickCount64;
            float[][] vocals = runner.Separate(backend, left, right, request.Overlap, progress: LogProgress, cancel: ct);
            Logs.Verbose($"[Audio][Kim-Vocal-2] Separated {AudioClipCodec.Seconds(left.Length, MdxSettings.KimVocal2.SampleRate):0.0}s on {backend.Device} in {Environment.TickCount64 - started}ms.");
            return new StemsResult
            {
                Stems = new Dictionary<string, byte[]>(StringComparer.Ordinal)
                {
                    ["vocals"] = AudioClipCodec.EncodeWav(vocals[0], vocals[1], runner.SampleRate),
                },
                Format = "wav",
                SampleRate = runner.SampleRate,
            };
        }, cancel);
    }

    /// <summary>The vocals of <paramref name="audio"/> for the video driving-audio pre-step, in both forms: a 44.1 kHz stereo WAV clip (16-bit PCM) for a soundtrack or a recipe that decodes it, and 16 kHz mono taken from the float stem the way the reference does, for a speech encoder.</summary>
    internal Task<SeparatedVocals> SeparateVocalsClipAsync(AudioClip audio, double? maxSeconds, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(audio);
        AudioModelSelector selector = new(FxCatalog.KimVocal2Id, FxCatalog.KimVocal2Id, null);
        // CPU on purpose: this runs just before a video model that needs the card, and no request with SeparateDrivingVocals has run on a GPU.
        IBackend backend = _cpuBackend.Value;
        return _engine.AudioRuntime.RunAsync(backend, new AudioJob(_engine.AudioRuntime.Mdx, FxCatalog.KimVocal2Id), async ct =>
        {
            (float[] left, float[] right) = AudioClipCodec.DecodeStereoSoxrHq(audio, MdxSettings.KimVocal2.SampleRate);
            if (left.Length == 0)
            {
                throw new ArgumentException("The driving audio decoded to no samples.", nameof(audio));
            }
            MdxVocalRunner runner = await LoadKimVocal2Async(selector, FxCatalog.KimVocal2Id, ct).ConfigureAwait(false);
            float[][] vocals = runner.Separate(backend, left, right, null, maxSeconds, LogProgress, ct);
            AudioClip stem = new() { Data = AudioClipCodec.EncodeWav(vocals[0], vocals[1], runner.SampleRate), Format = "wav" };
            // Uncropped: the recipe cuts it to the length of the clip it actually generates, which a chain rounds up.
            return new SeparatedVocals(stem, MdxSeparator.ToMono(vocals, runner.SampleRate, SeparatedVocals.SpeechSampleRate));
        }, cancel);
    }

    private Task<MdxVocalRunner> LoadKimVocal2Async(AudioModelSelector selector, string cacheKey, CancellationToken cancel) =>
        _engine.AudioRuntime.Mdx.GetOrLoadAsync(cacheKey, async ct =>
        {
            string path = await FxCatalog.EnsureKimVocal2PathAsync(selector.LocalPath, ct).ConfigureAwait(false);
            return FxCatalog.LoadKimVocal2(path);
        }, cancel);

    private static void LogIsolatorProgress(int done, int total) => Logs.Verbose($"[Audio][Voice-Isolator] window {done}/{total}");

    private static void LogProgress(int done, int total) => Logs.Verbose($"[Audio][Kim-Vocal-2] chunk {done}/{total}");

    private static void RejectDemucsOnlyKnobs(FxSeparateRequest request)
    {
        if (request.Shifts != 0 || request.Segment is not null)
        {
            throw new ArgumentException("Shifts and Segment are Demucs options; Kim_Vocal_2 takes only Overlap.", nameof(request));
        }
        if (request.Overlap is < 0 or >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Overlap must be in [0, 1).");
        }
    }

    /// <inheritdoc/>
    public Task<AudioResult> EnhanceAsync(ModelSpec spec, FxEnhanceRequest request, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        // Unlike Demucs, Resemble-Enhance's STFT (ResembleDenoiser.cs / ResembleEnhancePipeline.cs, both call
        // Fft.RealTransform directly) never touches IBackend.Stft — only its neural compute (UNet/WN/UnivNet)
        // goes through backend, using ops CUDA already implements — so it can run on the engine's real backend.
        // Verified 2026-08-09: this does NOT fix the known >10min stall on a short clip — a single host thread
        // pins at 100% CPU with near-zero GPU utilization throughout, meaning the bottleneck is host-side DSP
        // (very likely the hand-rolled Fft.RealTransform loop itself, or per-frame NSF/harmonic-source work,
        // none of which this change touches), not the neural compute this moves to GPU. Root cause still open.
        IBackend backend = _engine.Backend;

        return _engine.AudioRuntime.RunAsync(backend, new AudioJob(_engine.AudioRuntime.Enhance, FxCatalog.EnhanceRepo), async ct =>
        {
            float[] mono = AudioClipCodec.DecodeMono(request.Audio, FxCatalog.EnhanceSampleRate);
            if (mono.Length == 0)
            {
                throw new ArgumentException("The audio to enhance decoded to no samples.", nameof(request));
            }
            ct.ThrowIfCancellationRequested();

            EnhanceRunner runner = await _engine.AudioRuntime.Enhance
                .GetOrLoadAsync(FxCatalog.EnhanceRepo, FxCatalog.LoadEnhanceAsync, ct).ConfigureAwait(false);
            long started = Environment.TickCount64;
            // lambd = denoise/enhance blend, tau = prior temperature, seed = determinism. The engine's Enhance does
            // not take an LCFM step count or solver yet, so those knobs have no counterpart here.
            float lambd = (float)(request.Lambd ?? 0.5);
            float tau = (float)(request.Tau ?? 0.5);
            float[] enhanced = request.DenoiseOnly ? runner.Denoise(backend, mono)
                : runner.Enhance(backend, mono, lambd, tau, request.Seed < 0 ? 0 : request.Seed, request.Nfe, request.Solver);
            double seconds = AudioClipCodec.Seconds(enhanced.Length, runner.SampleRate);
            Logs.Verbose($"[Audio][Resemble-Enhance] Enhanced {AudioClipCodec.Seconds(mono.Length, FxCatalog.EnhanceSampleRate):0.0}s "
                + $"in {Environment.TickCount64 - started}ms.");
            return new AudioResult
            {
                Data = AudioClipCodec.EncodeWav(enhanced, null, runner.SampleRate),
                Format = "wav",
                DurationSeconds = seconds,
                SampleRate = runner.SampleRate,
                Meta = new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["model"] = FxCatalog.EnhanceRepo,
                    ["seed"] = request.Seed.ToString(CultureInfo.InvariantCulture),
                },
            };
        }, cancel);
    }
}
