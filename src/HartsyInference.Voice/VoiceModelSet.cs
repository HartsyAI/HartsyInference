using System.Diagnostics;
using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Audio.Models.Wake;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio.Wake;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Voice.Audio;
using HartsyInference.Voice.Gpu;

namespace HartsyInference.Voice;

/// <summary>The models every call shares, loaded once for the host's lifetime: the speech recognizer and synthesizer
/// on the audio device, run only by this set's GPU thread, and the VAD and denoiser weights each session instantiates
/// its own state over.</summary>
/// <remarks>The set owns the GPU thread (T2), so every session's recognition and synthesis queue behind one another
/// on the one thread allowed to touch the audio device, and it is disposed on that thread too. When
/// <see cref="VoiceAgentOptions.CpuThreadCap"/> is positive the engine's CPU kernel threads are capped while the set
/// is loaded and the setting found at load is restored on dispose. The knob is process-wide, so with two capped sets
/// alive at once, dispose them in reverse load order or the earlier cap outlives both.</remarks>
public sealed class VoiceModelSet : IAsyncDisposable
{
    private readonly IVoiceSpeech _speech;
    private readonly Func<IVadModel> _createVad;
    private readonly Func<RnnoiseStream>? _createDenoiser;
    private readonly IDisposable? _frontEndModels;
    private readonly bool _cpuCapApplied;
    private readonly bool _cpuCapHadOverride;
    private readonly int _cpuCapPrevious;
    private int _disposed;

    /// <summary>Wraps loaded models. Takes ownership of <paramref name="speech"/> (disposed on the GPU thread) and
    /// <paramref name="frontEndModels"/>; <paramref name="audioDevice"/> is borrowed.</summary>
    internal VoiceModelSet(VoiceAgentOptions options, IVoiceSpeech speech, IBackend audioDevice, Func<IVadModel> createVad,
        Func<RnnoiseStream>? createDenoiser, IDisposable? frontEndModels = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(speech);
        ArgumentNullException.ThrowIfNull(audioDevice);
        ArgumentNullException.ThrowIfNull(createVad);
        options.Validate();
        if (options.Denoise && createDenoiser is null)
        {
            throw new ArgumentException("Denoise is on but no denoiser was supplied.", nameof(createDenoiser));
        }
        Options = options;
        _speech = speech;
        _createVad = createVad;
        _createDenoiser = options.Denoise ? createDenoiser : null;
        _frontEndModels = frontEndModels;
        if (options.CpuThreadCap > 0)
        {
            // Reading the value first loads the settings file, so a file value cannot later overwrite this override.
            _cpuCapPrevious = EngineKnobs.CpuThreads.Value;
            _cpuCapHadOverride = KnobStore.HasOverride(EngineKnobs.CpuThreads);
            KnobStore.Set(EngineKnobs.CpuThreads, options.CpuThreadCap);
            _cpuCapApplied = true;
        }
        // Last, so nothing after it can fail and leave the thread running.
        Gpu = new VoiceGpuWorker(audioDevice, speech.Reopen);
    }

    /// <summary>The options the set was loaded with; sessions on it must carry the same model and device fields.</summary>
    public VoiceAgentOptions Options { get; }

    /// <summary>Whether sessions run RNNoise ahead of the VAD.</summary>
    public bool DenoiseEnabled => _createDenoiser is not null;

    /// <summary>The GPU thread; the only caller of the speech models.</summary>
    internal VoiceGpuWorker Gpu { get; }

    /// <summary>Rate of synthesized audio.</summary>
    internal int SynthesisSampleRate => _speech.SynthesisSampleRate;

    /// <summary>Recognizes one utterance; GPU thread only.</summary>
    internal string Transcribe(float[] audio) => _speech.Transcribe(audio);

    /// <summary>Synthesizes one sentence; GPU thread only.</summary>
    internal float[] Synthesize(string text) => _speech.Synthesize(text);

    /// <summary>A VAD instance for one session; it carries that session's recurrent state.</summary>
    internal IVadModel CreateVadModel() => _createVad();

    /// <summary>A denoiser for one session, or null when denoising is off.</summary>
    internal RnnoiseStream? CreateDenoiser() => _createDenoiser?.Invoke();

    /// <summary>Opens the speech models on <paramref name="engine"/> and loads the front-end weights.</summary>
    /// <param name="engine">Built on <see cref="VoiceAgentOptions.AudioDevice"/>; the language model reaches another card
    /// through <see cref="VoiceAgentOptions.LlmDevice"/> on each request. It must outlive the set. An engine release
    /// (free memory, backend switch) revokes the speech leases; the GPU thread reopens them on the next job that finds
    /// them gone and retries it once.</param>
    /// <param name="options">Models, devices and front-end settings.</param>
    /// <param name="wakeModelRoot">Folder holding <c>vad</c> and <c>denoise</c>; defaults to the models root's
    /// <c>audio/wake</c>.</param>
    /// <param name="cancel">Stops the load.</param>
    public static async Task<VoiceModelSet> LoadAsync(InferenceEngine engine, VoiceAgentOptions options, string? wakeModelRoot = null,
        CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        IBackend device = engine.ComputeBackend;
        string deviceKey = device.Device.ToString();
        if (!string.Equals(deviceKey, options.AudioDevice, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"The engine runs on {deviceKey} but AudioDevice is {options.AudioDevice}. Build the engine on the audio "
                + "device and reach the language model's card with LlmDevice.", nameof(engine));
        }
        // The small front-end files first, so a missing VAD or denoiser fails before the speech models load.
        WakeModelSet frontEnd = LoadFrontEnd(wakeModelRoot ?? Path.Combine(RepoPaths.ModelsRoot(), "audio", "wake"), options,
            out Func<IVadModel> createVad, out Func<RnnoiseStream>? createDenoiser);
        VoiceLeaseSpeech? speech = null;
        try
        {
            ModelSpec tts = ModelResolver.Resolve(options.TtsModel, null, Modality.Speech);
            ModelSpec stt = ModelResolver.Resolve(options.SttModel, null, Modality.Transcribe);
            speech = await VoiceLeaseSpeech.OpenAsync(token => engine.Speech.OpenSynthesizerAsync(tts, token),
                token => engine.Transcribe.OpenTranscriberAsync(stt, token), ModelSelector.Parse(options.TtsModel).Variant, cancel).ConfigureAwait(false);
            return new VoiceModelSet(options, speech, device, createVad, createDenoiser, frontEnd);
        }
        catch
        {
            speech?.Dispose();
            frontEnd.Dispose();
            throw;
        }
    }

    /// <summary>Loads the per-session front-end weights from <paramref name="wakeModelRoot"/> (the wake models'
    /// <c>vad</c> and <c>denoise</c> folders). Silero is required; RNNoise is required when
    /// <see cref="VoiceAgentOptions.Denoise"/> is on, and its absence fails here rather than degrading to raw audio.</summary>
    internal static WakeModelSet LoadFrontEnd(string wakeModelRoot, VoiceAgentOptions options, out Func<IVadModel> createVad,
        out Func<RnnoiseStream>? createDenoiser)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(wakeModelRoot);
        ArgumentNullException.ThrowIfNull(options);
        WakeModelSet wake = new(wakeModelRoot);
        try
        {
            if (options.Denoise && !wake.LoadDenoiser())
            {
                throw new FileNotFoundException(
                    $"Denoise is on but no usable RNNoise weights were found at '{Path.Combine(wakeModelRoot, "denoise", "rnnoise.safetensors")}'. "
                    + "Install them or set Denoise to false; the voice session never substitutes unprocessed audio for a missing denoiser.");
            }
            if (!wake.LoadVad())
            {
                throw new FileNotFoundException(
                    $"No usable Silero VAD weights under '{Path.Combine(wakeModelRoot, "vad")}' (silero_vad_16k.safetensors or silero_vad.onnx); "
                    + "the voice session cannot find the end of a caller's turn without them.");
            }
            int minSilenceMs = options.EndOfTurnSilenceMs;
            createVad = () => wake.CreateVad(minSilenceMs)?.Model
                ?? throw new InvalidOperationException($"Could not instantiate the Silero VAD from '{wakeModelRoot}'.");
            createDenoiser = options.Denoise
                ? () => wake.CreateDenoiser() ?? throw new InvalidOperationException($"Could not instantiate RNNoise from '{wakeModelRoot}'.")
                : null;
            return wake;
        }
        catch
        {
            wake.Dispose();
            throw;
        }
    }

    /// <summary>The language model's spec for <paramref name="options"/>.</summary>
    internal static ModelSpec ResolveLlm(VoiceAgentOptions options) => ModelResolver.Resolve(options.LlmModel, null, Modality.Text);

    /// <summary>What the warm-up synthesizes: texts of 1, 3, 6, 13 and 30 words. On the RTX 3060 they gave 1.45, 1.75,
    /// 2.03, 4.88 and 10.18 s of audio, so 58, 70, 81, 195 and 407 of Kokoro's 25 ms alignment frames. Those fall in
    /// the power-of-two buckets 64, 128, 128, 256 and 512, which are all the buckets a sentence reaches:
    /// <list type="bullet">
    /// <item>even "Okay." carries about a second of edge audio, so nothing lands in the 32 bucket;</item>
    /// <item><see cref="VoiceAgentOptions.MaxSentenceChars"/> keeps a sentence within the 512 bucket.</item>
    /// </list>
    /// Kokoro chooses its convolution plans per length bucket, so the first sentence a caller hears finds its bucket's
    /// plans already built. The 256 bucket holds the 15-word sentences.</summary>
    internal static IReadOnlyList<string> WarmTexts { get; } =
    [
        "Okay.",
        "Thanks for calling.",
        "Let me check that for you.",
        "Your order is on its way, and it should arrive by Thursday afternoon.",
        "I have updated the delivery address on your order, the driver will call you when they are ten minutes away, "
            + "and you will receive a message with the tracking link.",
    ];

    /// <summary>Loads every model into memory and builds its per-length state before the first call: one synthesis per
    /// text of <see cref="WarmTexts"/> and one recognition of a second of silence, each its own job on the GPU thread,
    /// and a one-token generation on the language model's device. Logs one <c>[Voice] Warm-up</c> line with each
    /// job's time on the GPU thread and the audio length of each synthesis.</summary>
    public async Task WarmAsync(ITextService text, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        float[] silence = new float[VoiceAudioFrontend.SampleRate];
        TextRequest request = new()
        {
            Messages =
            [
                new TextMessage { Role = TextRole.System, Content = Options.SystemPrompt },
                new TextMessage { Role = TextRole.User, Content = "Hello." },
            ],
            Device = Options.LlmDevice,
            EnableThinking = false,
            MaxTokens = 1,
            AlwaysFreeMemory = false,
        };
        // Different devices, so the language model warms while the GPU thread does.
        Task<TextResult> llm = text.GenerateAsync(ResolveLlm(Options), request, cancel);
        Task speech = WarmSpeechAsync(silence, cancel);
        await Task.WhenAll(llm, speech).ConfigureAwait(false);
    }

    private async Task WarmSpeechAsync(float[] silence, CancellationToken cancel)
    {
        // One job per text, each under its own gate hold, as a turn's sentences are. Each is timed inside its job, so
        // the log shows the synthesis alone and not the queue ahead of it.
        (double Ms, double Seconds)[] synthesized = new (double, double)[WarmTexts.Count];
        for (int i = 0; i < synthesized.Length; i++)
        {
            string sentence = WarmTexts[i];
            synthesized[i] = await Gpu.RunAsync(VoiceGpuJobKind.Warm, () =>
            {
                long started = Stopwatch.GetTimestamp();
                float[] audio = _speech.Synthesize(sentence);
                return (Stopwatch.GetElapsedTime(started).TotalMilliseconds, audio.Length / (double)_speech.SynthesisSampleRate);
            }, cancel).ConfigureAwait(false);
        }
        double recognitionMs = await Gpu.RunAsync(VoiceGpuJobKind.Warm, () =>
        {
            long started = Stopwatch.GetTimestamp();
            _speech.Transcribe(silence);
            return Stopwatch.GetElapsedTime(started).TotalMilliseconds;
        }, cancel).ConfigureAwait(false);
        Logs.Info($"[Voice] Warm-up on {Options.AudioDevice}: synthesized "
            + $"{string.Join(" / ", WarmTexts.Select(text => text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length))} words in "
            + $"{string.Join(" / ", synthesized.Select(run => $"{run.Ms:F1}"))} ms "
            + $"({string.Join(" / ", synthesized.Select(run => $"{run.Seconds:F2}"))} s of audio); "
            + $"recognized {silence.Length / (double)VoiceAudioFrontend.SampleRate:0.#} s of silence in {recognitionMs:F1} ms.");
    }

    /// <summary>Releases the speech models on the GPU thread, stops it, releases the front-end weights and restores the
    /// CPU thread setting. Dispose every session first.</summary>
    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        try
        {
            await Gpu.StopAsync(_speech.Dispose).ConfigureAwait(false);
        }
        finally
        {
            Gpu.Dispose();
            _frontEndModels?.Dispose();
            if (_cpuCapApplied)
            {
                if (_cpuCapHadOverride)
                {
                    KnobStore.Set(EngineKnobs.CpuThreads, _cpuCapPrevious);
                }
                else
                {
                    KnobStore.Clear(EngineKnobs.CpuThreads);
                }
            }
        }
    }
}
