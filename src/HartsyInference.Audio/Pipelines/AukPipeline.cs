using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Auk;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.QwenOmni;
using HartsyInference.Core.Backends;
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
        bool sequential = opts.SequentialResidency && !FitsResident(backend, hasAudio);

        using Tensor text = EncodeConditioning(backend, instruction, audio16k, sequential, cancel);
        cancel.ThrowIfCancellationRequested();
        using Tensor? refLatent = hasAudio ? EncodeReference(backend, refPcm, opts, sequential) : null;
        cancel.ThrowIfCancellationRequested();
        using Tensor latent = Sample(backend, text, refLatent, frames, opts, sequential, cancel);
        return Decode(backend, latent, sequential);
    }

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

    /// <summary>True when the device has enough free VRAM to hold every stage this call will touch resident at
    /// once, so the caller can skip freeing between stages. Generous on purpose (every stage's full weight size,
    /// plus a third again for activations) — this only ever turns a requested sequential run into a resident one,
    /// never the other way around, so overestimating costs a redundant free/reload rather than an OOM.</summary>
    private bool FitsResident(IBackend backend, bool hasAudio)
    {
        (long freeBytes, long totalBytes) = backend.GetVramInfo();
        long bytes = WeightBytes(_lm.EnumerateWeights()) + WeightBytes(_dit.EnumerateWeights()) + WeightBytes(_vae.EnumerateWeights());
        if (hasAudio)
        {
            bytes += WeightBytes(_tower.EnumerateWeights()) + WeightBytes(_vaeEncoder!.EnumerateWeights());
        }
        return ResidentWithinBudget(freeBytes, totalBytes, bytes);
    }

    /// <summary>True when <paramref name="freeBytes"/> covers <paramref name="requiredBytes"/> plus a third again
    /// for activations. <paramref name="totalBytes"/> not being positive means the backend can't report VRAM at
    /// all (the CPU backend, mainly), which this treats as "assume the worst" rather than as boundless free memory.</summary>
    public static bool ResidentWithinBudget(long freeBytes, long totalBytes, long requiredBytes) =>
        totalBytes > 0 && freeBytes >= requiredBytes + requiredBytes / 3;

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
