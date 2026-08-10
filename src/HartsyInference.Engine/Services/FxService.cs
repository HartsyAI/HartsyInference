using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Audio-effects service: Demucs stem separation from a placed checkpoint, and Resemble-Enhance speech
/// enhancement, both run on the shared audio device under the generation lock.</summary>
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
        string modelName = string.IsNullOrWhiteSpace(request.Model) ? selector.Variant : request.Model;
        IBackend backend = _cpuBackend.Value;

        return _engine.AudioRuntime.RunAsync(backend, $"fx:demucs:{modelName}", async ct =>
        {
            (float[] left, float[] right) = AudioClipCodec.DecodeStereo(request.Audio, FxCatalog.DemucsSampleRate);
            if (left.Length == 0)
            {
                throw new ArgumentException("The audio to separate decoded to no samples.", nameof(request));
            }
            ct.ThrowIfCancellationRequested();

            string path = await FxCatalog.EnsureDemucsPathAsync(modelName, selector.LocalPath, ct).ConfigureAwait(false);
            DemucsRunner runner = await _engine.AudioRuntime.Demucs
                .GetOrLoadAsync(path, _ => Task.FromResult(FxCatalog.LoadDemucs(path, modelName)), ct).ConfigureAwait(false);
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

        return _engine.AudioRuntime.RunAsync(backend, $"fx:enhance:{FxCatalog.EnhanceRepo}", async ct =>
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
            float[] enhanced = runner.Enhance(backend, mono, lambd, tau, request.Seed < 0 ? 0 : request.Seed,
                request.Nfe, request.Solver);
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
