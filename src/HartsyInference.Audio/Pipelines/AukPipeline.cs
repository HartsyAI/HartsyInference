using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Auk;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.QwenOmni;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Tencent AuK / AuK-Flash: an instruction plus an optional audio clip steers a flow-matching DiT, vocoded by the AuK VAE; the task (clone, instruct TTS, edit, enhance, separate) lives entirely in the instruction text.</summary>
/// <remarks>
/// <code>
///   mel16k   = QwenOmni mel(audio @ 16 kHz)  ->  audio tower  ->  N tokens
///   ids      = ChatML(instruction, N x AUDIO)  ->  thinker embeds with the audio rows spliced
///   text     = layer fusion of the thinker's 36 hidden states                    # [1, nt, 2048]
///   ref      = VAE encoder(audio @ 24 kHz trimmed to whole hops)                 # [1, R, 64]
///   x        = N(0, 1) [1, frames, 64];  Euler over the schedule with [text | ref | x] DiT forwards (+ CFG on base)
///   pcm      = VAE decoder(denormalize(x))
/// </code>
/// Stages are sequenced so the audio tower, the thinker, the DiT and the VAE decoder are never device-resident together (<see cref="AukOptions.SequentialResidency"/>).
/// </remarks>
public sealed class AukPipeline : IAudioPipeline, IDisposable
{
    /// <summary>Longest reference clip or generated clip, in seconds.</summary>
    public const double MaxSeconds = 30.0;

    private readonly AukConfig _cfg;
    private readonly AukVaeConfig _vaeCfg;
    private readonly Qwen2Config _lmCfg;
    private readonly QwenOmniConfig _omniCfg;
    private readonly Func<string, IReadOnlyList<int>> _encode;
    private readonly bool _flash;
    private readonly AukDit _dit;
    private readonly AukVae _vae;
    private readonly Qwen2Model _lm;
    private readonly QwenOmniAudioEncoder _tower;
    private readonly QwenOmniProcessor _processor;
    private IReadOnlyDictionary<string, Tensor>? _checkpoint;
    private AukVaeEncoder? _vaeEncoder;
    private AukVaeStats? _stats;
    private bool _loaded;
    private int _disposed;

    /// <summary>Bytes this pipeline itself currently holds resident on the device (0 when nothing is kept warm
    /// between stages). Re-added to the live free-VRAM reading in <see cref="FitsResident"/> so the check reflects
    /// what is actually available rather than being confounded by this pipeline's own prior allocation — a plain
    /// re-read of <see cref="IBackend.GetVramInfo"/> after going resident would see "only free_before - mine" and
    /// flip straight back to evicting. Tracking it instead of caching the decision outright keeps this responsive
    /// to real external pressure (another engine's pipeline consuming VRAM in the meantime still lowers the live
    /// reading and can correctly flip this back to sequential).</summary>
    private long _residentBytes;

    /// <summary>The backend <see cref="_residentBytes"/> was measured against. A pipeline is only ever driven by
    /// one device in practice, but if a later call ever did pass a different <see cref="IBackend"/>, the byte count
    /// would describe VRAM on a device nobody is asking about -- so a mismatch is treated as a cold start on the
    /// new backend (and whatever this pipeline still holds on the old one is freed there, since nothing will ever
    /// account for it again otherwise).</summary>
    private IBackend? _residentBackend;

    /// <summary>Set once an actual OOM (not staleness, a real driver refusal while resident) is observed during
    /// <see cref="Generate"/>. <c>AudioRuntime.EvictForRetry</c>-style recovery wipes the device and retries
    /// the same call once; without this, the retry's fresh <see cref="FitsResident"/> measurement would very likely
    /// pass again (the sweep just freed everything) and repeat the exact same OOM with no way out. Once set, this
    /// pipeline instance never attempts resident mode again -- sequential is always safe, just slower.</summary>
    private bool _residentDisabledByOom;

    /// <inheritdoc/>
    public string ModelName { get; }

    /// <summary>True for AuK-Flash, which pins 4 steps and no guidance.</summary>
    public bool IsFlash => _flash;

    /// <summary>Output (and reference) sample rate in Hz.</summary>
    public int SampleRate => _cfg.SampleRate;

    /// <summary>Creates the pipeline without weights; <paramref name="encode"/> is the byte-level-exact Qwen2.5-Omni text encoder (control tokens are spliced by id, never scanned for).</summary>
    public AukPipeline(string modelName, bool flash, Func<string, IReadOnlyList<int>> encode, AukConfig? config = null,
        AukVaeConfig? vaeConfig = null, Qwen2Config? lmConfig = null, QwenOmniConfig? omniConfig = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelName);
        ArgumentNullException.ThrowIfNull(encode);
        _cfg = config ?? AukConfig.Default;
        _vaeCfg = vaeConfig ?? new AukVaeConfig();
        _lmCfg = lmConfig ?? Qwen2Config.Qwen25Omni_3B_Thinker;
        _omniCfg = omniConfig ?? QwenOmniConfig.Default;
        _vaeCfg.Validate();
        if (_vaeCfg.LatentDim != _cfg.LatentDim || _vaeCfg.Hop != _cfg.Hop || _vaeCfg.SampleRate != _cfg.SampleRate)
            throw new ArgumentException("The VAE latent width, hop and sample rate must match the DiT configuration.");
        if (_lmCfg.HiddenSize != _cfg.TextDim || _omniCfg.OutputDim != _lmCfg.HiddenSize || _lmCfg.NumHiddenLayers != _cfg.FusionLayers)
            throw new ArgumentException("The thinker hidden size, tower output width and fusion layer count must match the DiT text conditioning.");
        ModelName = modelName;
        _flash = flash;
        _encode = encode;
        _dit = new AukDit(_cfg);
        _vae = new AukVae(_vaeCfg);
        _lm = new Qwen2Model(_lmCfg);
        _tower = new QwenOmniAudioEncoder(_omniCfg);
        _processor = new QwenOmniProcessor(_omniCfg);
    }

    /// <summary>Loads the three checkpoints: <paramref name="auk"/> (<c>transformer.*</c>, <c>layer_weights</c>, <c>layer_scale</c>), <paramref name="vae"/> (decoder, <c>audio_encoder.*</c>, latent statistics) and <paramref name="omni"/> (<c>thinker.model.*</c>, <c>thinker.audio_tower.*</c>). The dictionaries' owners must outlive the pipeline.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> auk, IReadOnlyDictionary<string, Tensor> vae,
        IReadOnlyDictionary<string, Tensor> omni)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(auk);
        ArgumentNullException.ThrowIfNull(vae);
        ArgumentNullException.ThrowIfNull(omni);
        _dit.LoadWeights(auk);
        _vae.LoadWeights(vae);
        _vaeEncoder?.Dispose();
        _stats?.Dispose();
        _stats = AukVaeStats.Load(vae);
        _vaeEncoder = new AukVaeEncoder(_vaeCfg, _stats);
        _vaeEncoder.LoadWeights(vae);
        _lm.LoadWeightsNoLmHead(omni, "thinker.model");
        _tower.LoadWeights(omni);
        AukOps.Require(auk, AukLayerFusion.LayerWeightsKey, _cfg.FusionLayers);
        AukOps.Require(auk, AukLayerFusion.LayerScaleKey, 1);
        _checkpoint = auk;
        _loaded = true;
    }

    /// <summary>Generates mono PCM at <see cref="SampleRate"/>. <paramref name="audio"/> (mono, <paramref name="audioSampleRate"/> Hz) is the voice reference or the source to edit; null runs text-only instruct TTS and needs <paramref name="durationSeconds"/>. With audio and no duration the output is as long as the clip (or scaled by the text-length heuristic).</summary>
    public float[] Generate(IBackend backend, string instruction, float[]? audio, int audioSampleRate, double? durationSeconds,
        AukOptions? options = null, CancellationToken cancel = default)
    {
        ThrowIfDisposed();
        if (!_loaded) throw new InvalidOperationException("Call LoadWeights first.");
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentException.ThrowIfNullOrWhiteSpace(instruction);
        AukOptions opts = options ?? new AukOptions();
        bool hasAudio = audio is { Length: > 0 };
        if (hasAudio && audioSampleRate < 1) throw new ArgumentOutOfRangeException(nameof(audioSampleRate));

        float[] refPcm = [];
        float[] audio16k = [];
        double? refSeconds = null;
        if (hasAudio)
        {
            refSeconds = audio!.Length / (double)audioSampleRate;
            if (refSeconds > MaxSeconds) throw new ArgumentException($"AuK handles at most {MaxSeconds:0} s of audio; the clip is {refSeconds:0.0} s.", nameof(audio));
            refPcm = PrepareReference(audio!, audioSampleRate, _cfg.SampleRate, _cfg.Hop);
            audio16k = audioSampleRate == _omniCfg.SampleRate ? audio! : Resampler.Create(audioSampleRate, _omniCfg.SampleRate).Resample(audio!);
        }
        double seconds = AukDuration.ResolveSeconds(durationSeconds, opts.GenText, opts.RefText, refSeconds, opts.Speed);
        int frames = ResolveFrames(seconds, _cfg.SampleRate, _cfg.Hop);
        cancel.ThrowIfCancellationRequested();

        // opts.SequentialResidency=true is a REQUEST to keep peak VRAM low, not a mandate to re-upload every
        // stage's weights on every call: freeing a stage only matters when the device doesn't have room for the
        // next one too. Downgrading to "preload everything, free nothing" when there is clearly enough free VRAM
        // turns a full host->device re-upload of every stage (tower, thinker, DiT, VAE) into a one-time cost for
        // back-to-back calls on the same pipeline, which is the common case once a model is warm.
        //
        // opts.SequentialResidency=false is the opposite, explicit request -- bypass the heuristic (and, with it,
        // the one-time _residentDisabledByOom trip from a past failure) and go resident unconditionally. Bytes are
        // still recorded on this path: RunStage will not free anything behind this call (release=false throughout),
        // so without recording them here the next auto-detected call would under-count what the device truly holds
        // and could wrongly believe there is room to also go resident on top of it.
        bool sequential;
        if (opts.SequentialResidency)
        {
            sequential = !FitsResident(backend, hasAudio);
        }
        else
        {
            ReconcileResidentBackend(backend);
            _residentBytes = ComputeResidentBytes(hasAudio);
            sequential = false;
        }

        try
        {
            using Tensor text = EncodeConditioning(backend, instruction, audio16k, sequential, cancel);
            cancel.ThrowIfCancellationRequested();
            using Tensor? refLatent = hasAudio ? EncodeReference(backend, refPcm, opts, sequential) : null;
            cancel.ThrowIfCancellationRequested();
            using Tensor latent = Sample(backend, text, refLatent, frames, opts, sequential, cancel);
            return Decode(backend, latent, sequential);
        }
        catch (Exception ex)
        {
            // FitsResident already counted these bytes as resident, optimistically, before any stage actually
            // ran. A failure here leaves what is really on the device unknown: maybe nothing from this call
            // landed, maybe some stages did before a later one threw (cancellation after a successful preload,
            // most notably -- not just an OOM), and a caller's OOM recovery (AudioRuntime.EvictForRetry) may
            // already be wiping the whole backend's device memory on its way to retrying. Rather than leave that
            // ambiguity for the next call to guess at, make it true: free every stage now (a no-op for whatever
            // a sweep already took, or was never actually loaded) so our now-zeroed count matches reality instead
            // of merely hoping a future PreloadWeights redoes what in fact never got freed.
            _residentBytes = 0;
            FreeAllStageWeights(backend);
            // A genuine OOM is only resident mode's fault if this call was actually attempting it -- a sequential
            // call OOMing under temporary external pressure says nothing about the resident margin being wrong,
            // and permanently disabling resident for that would cost every later call the optimization for no
            // reason. When it *was* a resident attempt, the margin was wrong for real activation/allocator
            // overhead, not merely stale accounting -- and the caller's one-shot retry (AudioRuntime.EvictForRetry)
            // wipes the whole device first, so a fresh FitsResident measurement on that retry would very likely
            // pass again and reproduce the exact same OOM with no way out. Disabling resident mode for the rest of
            // this instance's life trades a permanently-slower pipeline for one that never gets stuck repeating a
            // failure it cannot recover from.
            if (ShouldDisableResidentAfterFailure(sequential, ex)) _residentDisabledByOom = true;
            throw;
        }
    }

    /// <summary>True if <paramref name="error"/> or anything in its inner-exception chain is an
    /// <see cref="OutOfVramException"/> -- the same signal <c>AudioRuntime.EvictForRetry</c> keys its one-shot
    /// retry off of, read locally since this pipeline has no reference to the engine project that defines that logic.</summary>
    public static bool IsOutOfVram(Exception error)
    {
        for (Exception? e = error; e is not null; e = e.InnerException)
        {
            if (e is OutOfVramException) return true;
        }
        return false;
    }

    /// <summary>True only for an out-of-VRAM failure during a call that was itself attempting resident mode
    /// (<paramref name="sequential"/> false). A sequential call OOMing says nothing about the resident margin being
    /// wrong -- it never went resident -- so permanently disabling resident for the rest of this instance's life
    /// over that would cost every later call the optimization for a failure resident mode had no part in.</summary>
    public static bool ShouldDisableResidentAfterFailure(bool sequential, Exception error) => !sequential && IsOutOfVram(error);

    /// <summary>Latent frames for <paramref name="seconds"/> of output, rejecting more than <see cref="MaxSeconds"/>.</summary>
    public static int ResolveFrames(double seconds, int sampleRate, int hop)
    {
        if (seconds > MaxSeconds) throw new ArgumentException($"AuK generates at most {MaxSeconds:0} s per call; {seconds:0.0} s was requested.", nameof(seconds));
        return AukDuration.Frames(seconds, sampleRate, hop);
    }

    /// <summary>Mono reference at <paramref name="targetRate"/> trimmed to a whole number of <paramref name="hop"/>-sample latent frames, as the encoder's frame count is only exact then.</summary>
    public static float[] PrepareReference(ReadOnlySpan<float> audio, int sampleRate, int targetRate, int hop)
    {
        float[] pcm = sampleRate == targetRate ? audio.ToArray() : Resampler.Create(sampleRate, targetRate).Resample(audio);
        int whole = pcm.Length / hop * hop;
        if (whole < hop) throw new ArgumentException($"The reference is shorter than one latent frame ({hop} samples at {targetRate} Hz).", nameof(audio));
        return whole == pcm.Length ? pcm : pcm[..whole];
    }

    /// <summary>Standard-normal initial latent <c>[1, frames, latentDim]</c> from <see cref="DeterministicRng"/>, same seed same noise.</summary>
    public static Tensor BuildNoise(int frames, int latentDim, int seed)
    {
        Tensor noise = new(new TensorShape(1, frames, latentDim), DType.F32);
        Span<float> span = noise.AsSpan<float>();
        uint state = DeterministicRng.Seed(seed);
        for (int i = 0; i < span.Length; i++) span[i] = DeterministicRng.NextGaussian(ref state);
        return noise;
    }

    /// <summary>The sampling schedule: AuK-Flash always its fixed grid, the base model from the options.</summary>
    public AukSchedule BuildSchedule(AukOptions options) =>
        _flash ? AukSchedule.Flash() : AukSchedule.Base(options.Steps ?? 32, options.CfgScale ?? 2f, options.SwayCoef);

    private Tensor EncodeConditioning(IBackend backend, string instruction, float[] audio16k, bool sequential, CancellationToken cancel)
    {
        Tensor? audioRows = null;
        int audioTokens = 0;
        if (audio16k.Length > 0)
        {
            audioRows = RunStage(backend, _tower.EnumerateWeights(), sequential, () => EncodeAudioTokens(backend, audio16k));
            audioTokens = (int)audioRows.Shape[0];
        }
        try
        {
            cancel.ThrowIfCancellationRequested();
            int[] ids = AukPrompt.BuildIds(instruction, audioTokens, _encode, _omniCfg);
            return RunStage(backend, _lm.EnumerateWeights(), sequential, () => RunThinker(backend, ids, audioRows));
        }
        finally
        {
            audioRows?.Dispose();
        }
    }

    private Tensor EncodeAudioTokens(IBackend backend, float[] audio16k)
    {
        using Tensor mel = new(new TensorShape(_omniCfg.NumMelBins, _processor.PaddedFrames), DType.F32);
        int frames = _processor.ComputeMel(audio16k, mel);
        return _tower.Forward(backend, mel, frames);
    }

    private Tensor RunThinker(IBackend backend, int[] ids, Tensor? audioRows)
    {
        int t = ids.Length;
        using Tensor embeds = new(new TensorShape(1, t, _lmCfg.HiddenSize), DType.F32);
        _lm.EmbedLookup(embeds, ids, 1, t);
        if (audioRows is not null)
        {
            QwenOmniProcessor.SpliceAudioRows(ids, _omniCfg.AudioTokenId, embeds.AsSpan<float>(), audioRows.AsSpan<float>(), _lmCfg.HiddenSize);
        }
        using AukLayerFusion fusion = AukLayerFusion.FromCheckpoint(backend, _checkpoint!, _cfg.TextDim);
        fusion.Begin(t);
        using IKvCache cache = _lm.CreateDecodeCache(t);
        using Tensor finalNormed = _lm.ForwardEmbeds(backend, embeds, 1, t, 0, cache, layerTap: fusion.OnLayer);
        return fusion.Complete(finalNormed);
    }

    private Tensor EncodeReference(IBackend backend, float[] refPcm, AukOptions opts, bool sequential)
    {
        using Tensor pcm = new(new TensorShape(1, 1, refPcm.Length), DType.F32);
        refPcm.CopyTo(pcm.AsSpan<float>());
        AukVaeEncoder encoder = _vaeEncoder!;
        return RunStage(backend, encoder.EnumerateWeights(), sequential,
            () => encoder.Encode(backend, pcm, unchecked(opts.Seed + 1)));
    }

    /// <summary>True when the device has enough free VRAM, once this pipeline's own already-resident bytes are
    /// added back in, to hold every stage this <paramref name="hasAudio"/> configuration touches resident at once.
    /// Re-evaluated every call (see <see cref="_residentBytes"/> for why that stays correct instead of flip-flopping)
    /// so a genuine drop in external free VRAM — another pipeline loading on the same device — still downgrades to
    /// sequential instead of an indefinitely stale "yes".</summary>
    private bool FitsResident(IBackend backend, bool hasAudio)
    {
        if (_residentDisabledByOom) return false;
        ReconcileResidentBackend(backend);
        (long freeBytes, long totalBytes) = backend.GetVramInfo();
        if (WasSweptExternally(totalBytes - freeBytes, _residentBytes))
        {
            _residentBytes = 0;
        }
        long bytes = ComputeResidentBytes(hasAudio);
        bool fits = ResidentWithinBudget(freeBytes + _residentBytes, totalBytes, bytes);
        if (!fits && _residentBytes > 0)
        {
            // Downgrading after a previous call went resident. RunStage only frees a stage it actually runs, and
            // this call might not touch all of them -- a no-audio call after an audio call went resident never
            // visits the tower or VAE encoder at all, so without this they would stay resident, unaccounted and
            // unfreed, for as long as the process runs. Freeing everything up front costs nothing extra: whichever
            // stages this call's own RunStage calls free again moments later are no-ops the second time.
            FreeAllStageWeights(backend);
        }
        _residentBytes = fits ? bytes : 0;
        return fits;
    }

    /// <summary>Sum of what every stage this <paramref name="hasAudio"/> configuration touches would occupy
    /// resident on the device.</summary>
    private long ComputeResidentBytes(bool hasAudio)
    {
        long bytes = WeightBytes(_lm.EnumerateWeights()) + WeightBytes(_dit.EnumerateWeights()) + WeightBytes(_vae.EnumerateWeights());
        if (hasAudio)
        {
            bytes += WeightBytes(_tower.EnumerateWeights()) + WeightBytes(_vaeEncoder!.EnumerateWeights());
        }
        return bytes;
    }

    /// <summary>Frees every stage's weights on <paramref name="backend"/>, a safe over-free since
    /// <see cref="IBackend.FreeWeights"/> is a no-op for a stage it never loaded or already freed.</summary>
    private void FreeAllStageWeights(IBackend backend)
    {
        backend.FreeWeights(_lm.EnumerateWeights());
        backend.FreeWeights(_dit.EnumerateWeights());
        backend.FreeWeights(_vae.EnumerateWeights());
        backend.FreeWeights(_tower.EnumerateWeights());
        backend.FreeWeights(_vaeEncoder!.EnumerateWeights());
    }

    /// <summary>Resets the resident-bytes count to a cold start whenever <paramref name="backend"/> is not the one
    /// <see cref="_residentBytes"/> was last measured against -- freeing whatever this pipeline still believes it
    /// holds on the old backend first, since nothing will ever account for it there again otherwise. A single
    /// pipeline is only ever driven by one device in practice, so this is a cheap defensive guard rather than a
    /// path expected to trigger.</summary>
    private void ReconcileResidentBackend(IBackend backend)
    {
        if (ReferenceEquals(backend, _residentBackend)) return;
        if (_residentBackend is not null && _residentBytes > 0)
        {
            FreeAllStageWeights(_residentBackend);
        }
        _residentBytes = 0;
        _residentBackend = backend;
    }

    /// <summary>True when <paramref name="freeBytes"/> covers <paramref name="requiredBytes"/> plus a third again
    /// for activations. <paramref name="totalBytes"/> not being positive means the backend can't report VRAM at
    /// all (the CPU backend, mainly), which this treats as "assume the worst" rather than as boundless free memory.</summary>
    public static bool ResidentWithinBudget(long freeBytes, long totalBytes, long requiredBytes) =>
        totalBytes > 0 && freeBytes >= requiredBytes + requiredBytes / 3;

    /// <summary>True when the device shows less in-use memory than this pipeline believes it alone holds
    /// resident -- the tell that a backend-wide sweep this pipeline has no visibility into (such as
    /// <c>AudioRuntime.UnloadOthers</c> evicting a sibling model, which frees the whole device including this
    /// pipeline's own already-resident weights) ran since the bytes were counted. In-use memory can't be less
    /// than what we alone hold without at least ours having been freed too, so this needs no dedicated
    /// invalidation hook from the backend -- just the same <see cref="IBackend.GetVramInfo"/> call already made.</summary>
    public static bool WasSweptExternally(long usedBytes, long residentBytes) => usedBytes < residentBytes;

    /// <summary>Total device bytes <paramref name="weights"/> would occupy, element count times dtype size.</summary>
    public static long WeightBytes(IEnumerable<Tensor> weights)
    {
        long total = 0;
        foreach (Tensor w in weights) total += Tensor.ComputeByteSize(w.Shape, w.DType);
        return total;
    }

    private Tensor Sample(IBackend backend, Tensor text, Tensor? refLatent, int frames, AukOptions opts, bool sequential, CancellationToken cancel)
    {
        AukSchedule schedule = BuildSchedule(opts);
        return RunStage(backend, _dit.EnumerateWeights(), sequential, () =>
        {
            using Tensor condText = _dit.ProjectText(backend, text);
            using Tensor? uncondText = schedule.UsesCfg ? _dit.ProjectText(backend, text, drop: true) : null;
            Tensor x = BuildNoise(frames, _cfg.LatentDim, opts.Seed);
            try
            {
                for (int step = 0; step < schedule.Steps; step++)
                {
                    cancel.ThrowIfCancellationRequested();
                    float t = schedule.Timesteps[step];
                    using Tensor vCond = _dit.Forward(backend, x, refLatent, condText, t);
                    if (uncondText is not null)
                    {
                        using Tensor vUncond = _dit.Forward(backend, x, refLatent, uncondText, t, dropAudioCond: true);
                        AukSchedule.CfgCombine(vCond.AsSpan<float>(), vUncond.AsSpan<float>(), schedule.Cfg);
                    }
                    AukSchedule.EulerStep(x.AsSpan<float>(), vCond.AsSpan<float>(), schedule.Deltas[step]);
                }
                return x;
            }
            catch
            {
                x.Dispose();
                throw;
            }
        });
    }

    private float[] Decode(IBackend backend, Tensor latent, bool sequential)
    {
        using Tensor pcm = RunStage(backend, _vae.EnumerateWeights(), sequential, () =>
        {
            using Tensor raw = _stats!.Denormalize(backend, latent);
            return _vae.DecodeTimeMajor(backend, raw);
        });
        float[] samples = pcm.AsSpan<float>().ToArray();
        foreach (float v in samples)
        {
            if (!float.IsFinite(v)) throw new InvalidOperationException("AuK produced non-finite audio.");
        }
        return samples;
    }

    private static T RunStage<T>(IBackend backend, IEnumerable<Tensor> weights, bool release, Func<T> stage)
    {
        List<Tensor> held = [.. weights];
        backend.PreloadWeights(held);
        try
        {
            return stage();
        }
        finally
        {
            if (release) backend.FreeWeights(held);
        }
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(AukPipeline));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _dit.Dispose();
        _vae.Dispose();
        _vaeEncoder?.Dispose();
        _stats?.Dispose();
        _lm.Dispose();
        _tower.Dispose();
    }
}
