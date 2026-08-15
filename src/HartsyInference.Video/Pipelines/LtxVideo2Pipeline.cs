using System.Diagnostics;
using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>LTX-2.3 / LTX-2.5 (Lightricks, 22B) text-to-video+audio pipeline. Drives the dual-stream
/// <see cref="LtxVideo2Transformer"/> end-to-end: the all-49-layer text-tower features (Gemma-3-12B on 2.3,
/// Gemma-4-12B on 2.5) → per-modality <see cref="LtxVideo2TextConnectors"/> → flow-match Euler denoise of the
/// interleaved video+audio latent streams (2-way text CFG) → <see cref="LtxVideo2VaeDecoder"/> for RGB and
/// <see cref="LtxAudioVaeDecoder"/> + <see cref="LtxAudioVocoder"/> for the waveform.
///
/// <para>Token packing follows LTX patch-1: the video latent <c>[1,128,T,H,W]</c> packs to <c>[T·H·W, 128]</c> in
/// (f,h,w) order and the audio latent <c>[1,8,L,16]</c> packs to <c>[L, 128]</c> (channel·16+mel). The dual-stream
/// DiT consumes both each step and returns both velocities; CFG is standard velocity-space (the reference's
/// velocity→x0→delta→velocity round-trip reduces to this when guidance-rescale / STG / modality-isolation are off,
/// which are the defaults).</para></summary>
public sealed unsafe class LtxVideo2Pipeline : DiffusionPipelineBase
{
    private const int GemmaCaptionChannels = 3840;
    private const int GemmaLayers = 49;             // 48 transformer layers + 1 embedding
    private const int ConnectorRegisters = 128;     // text seq is padded to a multiple of this

    /// <summary>Shortest conditioning sequence to pad to, whatever the prompt length; 0 uses only the register
    /// multiple. LTX-2.5's Gemma 4 conditions at 1024; LTX-2.3 leaves this 0 and is unaffected.</summary>
    public int MinimumTextConditioningLength { get; init; }

    private readonly LtxVideo2Transformer _transformer;
    private readonly LtxVideo2TextConnectors _connectors;
    private readonly LtxVideo2VaeDecoder _vae;
    // LTX-2.5's diffusion video decoder, when the checkpoint ships it. Takes an ALREADY-un-normalized latent plus
    // injected noise, so the denormalization the conv decoder does internally happens at the call site instead.
    private readonly LtxVideo25DiffusionDecoder? _diffusionVae;
    private readonly float[]? _videoLatentsMean, _videoLatentsStd;
    private readonly LtxAudioVaeDecoder? _audioVae;
    private readonly LtxAudioVocoder? _vocoder;
    private readonly ILtx2TextTower _gemma;
    private readonly LtxVideo2Config _config;
    private readonly float[]? _audioLatentsMean, _audioLatentsStd;

    // Prompt-embedding cache (the FLite/Flux2 pattern): a hit skips the whole Gemma-3-12B encode + connector
    // phase, including the ~12 GB TE weight upload. The paired-CFG denoise consumes all four embeddings
    // (video/audio × pos/neg) every step, so they live under one (pos,neg) token key. The tensors are
    // host-materialized so they survive activation sweeps between generations.
    private int[]? _cachedPosKey;
    private int[]? _cachedNegKey;
    private Tensor? _cachedVideoPos, _cachedAudioPos, _cachedVideoNeg, _cachedAudioNeg;

    // Persistent resident prefix (the Flux KEEP_MODELS idiom, adapted to the streamed 22B DiT): the shared
    // weights + the first _residentPrefixBlocks blocks stay device-resident ACROSS generations, so the next
    // gen's PreloadWeights is a cache-hit no-op instead of a multi-GB re-upload — and the boundary the
    // streaming suffix starts at never moves (per-gen re-sizing drifted 12→10→9 blocks from pool slack).
    // The count is sized once and pinned; re-sized only when the token load grows past what the sizing
    // reserved headroom for. A prompt-cache MISS evicts the prefix when the ~12 GB Gemma encode doesn't fit
    // beside it (decided from measured free VRAM, logged either way).
    private static readonly bool KeepModelsResident = EnvSwitch.IsEnabled("HARTSY_KEEP_MODELS", defaultOn: true);
    private bool _prefixResident;
    // Set when the VAE decode freed only the prefix's tail: the pin still stands, so the next generation tops the
    // tail back up rather than re-uploading all 48 blocks — but it must trim first to hand the pool's decode
    // transients back to the allocator that re-upload draws on.
    private bool _prefixTailEvicted;
    private int _residentPrefixBlocks = -1;
    private long _prefixSizedTokens = -1;
    private long _gemmaWeightBytes = -1;

    public LtxVideo2Pipeline(IBackend backend, LtxVideo2Transformer transformer, LtxVideo2TextConnectors connectors,
        LtxVideo2VaeDecoder vae, ILtx2TextTower gemma, LtxVideo2Config config,
        LtxAudioVaeDecoder? audioVae = null, LtxAudioVocoder? vocoder = null,
        float[]? audioLatentsMean = null, float[]? audioLatentsStd = null,
        LtxVideo25DiffusionDecoder? diffusionVae = null,
        float[]? videoLatentsMean = null, float[]? videoLatentsStd = null)
        : base(backend)
    {
        _transformer = transformer;
        _connectors = connectors;
        _vae = vae;
        _gemma = gemma;
        _config = config;
        _audioVae = audioVae;
        _vocoder = vocoder;
        _audioLatentsMean = audioLatentsMean;
        _audioLatentsStd = audioLatentsStd;
        _diffusionVae = diffusionVae;
        _videoLatentsMean = videoLatentsMean;
        _videoLatentsStd = videoLatentsStd;
    }

    /// <summary>Result of a text-to-video generation: interleaved-RGB frames plus the (optional) decoded stereo
    /// 48 kHz waveform <c>[channels, samples]</c> in [-1, 1].</summary>
    public readonly record struct Ltx2Result(byte[][] Frames, int Width, int Height, int Seed,
        float[][]? Audio, int AudioSampleRate);

    /// <summary>Generates frames (and audio) from raw Gemma token id sequences. <paramref name="promptTokens"/> and
    /// <paramref name="negativeTokens"/> are single-prompt token id arrays; they are padded internally to a multiple
    /// of the connector register count. Set <paramref name="numFrames"/> so <c>(numFrames-1) % 8 == 0</c>.</summary>
    public Ltx2Result GenerateFromTokens(int[] promptTokens, int[] negativeTokens, TextToImageRequest request,
        int numFrames, double frameRate = 24.0, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        int width = request.Width ?? 768, height = request.Height ?? 512;
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        if (width % sp != 0 || height % sp != 0)
            throw new ArgumentException($"Width/height must be divisible by {sp} for LTX-2.");
        if (numFrames < 1 || (numFrames - 1) % tp != 0)
            throw new ArgumentException($"num_frames must satisfy (num_frames-1) % {tp} == 0; got {numFrames}.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int tLat = (numFrames - 1) / tp + 1;
        int hLat = height / sp, wLat = width / sp;
        int sv = tLat * hLat * wLat;
        int videoChannels = _config.InChannels;

        double durationS = numFrames / frameRate;
        double audioLatentsPerSecond = (double)_config.AudioSamplingRate / _config.AudioHopLength / _config.AudioScaleFactor;
        int audioFrames = Math.Max(1, (int)Math.Round(durationS * audioLatentsPerSecond));
        int audioChannels = _config.AudioInChannels;   // 8 latent ch × 16 mel-latent bins (patch-1 pack)

        int steps = request.Steps ?? _config.NumInferenceSteps;
        if (_config.FixedSigmas is { Length: > 1 } fixedSigmas)
        {
            int distilledSteps = fixedSigmas.Length - 1;
            if (steps != distilledSteps)
            {
                Logs.Warning($"LTX-2 distilled: ignoring the requested {steps} steps — this checkpoint was distilled " +
                    $"onto a fixed {distilledSteps}-step schedule, and any other count is a different schedule.");
                steps = distilledSteps;
            }
        }
        float guidance = request.CfgScale ?? _config.GuidanceScale;
        // The reference carries a separate audio CFG scale; unset follows the video scale.
        float audioGuidance = _config.AudioGuidanceScale ?? guidance;
        if (Environment.GetEnvironmentVariable("HARTSY_LTX2_AUDIO_CFG") is { Length: > 0 } audioCfgOverride
            && float.TryParse(audioCfgOverride, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsedAudioCfg))
        {
            audioGuidance = parsedAudioCfg;
        }
        // Both scales must be 1 to skip the unconditional branch: the single-branch path has no unconditional
        // velocity to give the audio Euler step, so a guided audio stream still needs the pair.
        bool unguided = guidance == 1f && audioGuidance == 1f;
        float audioRescale = _config.AudioGuidanceRescale;
        if (Environment.GetEnvironmentVariable("HARTSY_LTX2_AUDIO_RESCALE") is { Length: > 0 } rescaleOverride
            && float.TryParse(rescaleOverride, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsedRescale))
        {
            audioRescale = parsedRescale;
        }

        // Dynamic flow-match shift (LTX-2 scheduler: base_seq 1024 → base_shift 0.95, max_seq 4096 → max_shift 2.05).
        double m = (2.05 - 0.95) / (4096 - 1024), bShift = 0.95 - m * 1024;
        float shift = (float)Math.Exp(sv * m + bShift);

        Logs.Info($"LTX-2 T2V+A: {numFrames}f {width}x{height}, {steps} steps, cfg={guidance}, " +
            $"seed={seed} (video {tLat}x{hLat}x{wLat}={sv} tokens, audio {audioFrames} tokens, shift={shift:F3})");

        // 1. Text conditioning: Gemma 49-layer features → per-modality connector embeddings. Cached across
        // generations keyed on the token ids (the FLite/Flux2 prompt-cache pattern) — a hit skips the whole
        // Gemma phase including its ~12 GB weight upload.
        Stopwatch phase = Stopwatch.StartNew();
        bool cacheHit = _cachedPosKey is not null && _cachedNegKey is not null
            && promptTokens.AsSpan().SequenceEqual(_cachedPosKey)
            && negativeTokens.AsSpan().SequenceEqual(_cachedNegKey);
        Tensor encVideoPos, encAudioPos, encVideoNeg, encAudioNeg;
        if (cacheHit)
        {
            (encVideoPos, encAudioPos) = (_cachedVideoPos!, _cachedAudioPos!);
            (encVideoNeg, encAudioNeg) = (_cachedVideoNeg!, _cachedAudioNeg!);
            Logs.Info("[ltx2-phase] prompt cache HIT — skipping Gemma encode.");
        }
        else
        {
            // Reclaim pool slack from the previous generation before the ~12 GB Gemma upload.
            TextEncoderBackend.TrimMemoryPool();
            // The persistent resident prefix survives cache hits for free, but a MISS needs the Gemma weights
            // on device: keep the prefix only when the encoder actually fits beside it (measured free VRAM),
            // otherwise evict — the denoise section re-uploads at the pinned prefix size after the TE is freed.
            // With Gemma on its OWN device (TextEncoderBackend) it never contends with the prefix, so the whole
            // fit-check/evict dance is skipped and the resident DiT prefix survives every prompt miss.
            if (_prefixResident && ReferenceEquals(TextEncoderBackend, Backend))
            {
                if (_gemmaWeightBytes < 0)
                {
                    _gemmaWeightBytes = 0;
                    foreach (Tensor t in _gemma.EnumerateWeights())
                        _gemmaWeightBytes += t.DType.ComputeByteCount(t.ElementCount);
                }
                long freeNow = Backend.FreeMemoryBytes();
                long teNeed = _gemmaWeightBytes + (2L << 30);
                if (freeNow < teNeed)
                {
                    Backend.FreeWeights(_transformer.EnumerateWeights());
                    Backend.TrimMemoryPool();
                    _prefixResident = false;
                    Logs.Info($"[ltx2-phase] prompt MISS: evicted resident prefix for the Gemma encode " +
                        $"(free {freeNow >> 20} MB < TE {_gemmaWeightBytes >> 20} MB + 2048 MB margin).");
                }
                else
                {
                    Logs.Info($"[ltx2-phase] prompt MISS: Gemma fits beside the resident prefix " +
                        $"(free {freeNow >> 20} MB ≥ TE {_gemmaWeightBytes >> 20} MB + 2048 MB margin).");
                }
            }
            (encVideoPos, encAudioPos) = EncodeText(promptTokens);
            (encVideoNeg, encAudioNeg) = EncodeText(negativeTokens);

            // Reclaim the ~12 GB Gemma encoder AND the ~4 GB text connectors before the DiT — none are needed
            // during denoise (the connectors already produced the four cached embeddings). Freeing the connectors
            // is what lets the fp8 DiT (~18 GB) fit ALL 48 blocks resident on 24 GB (no streaming + graph-eligible);
            // a prompt-cache MISS transparently re-uploads them on the next EncodeText.
            TextEncoderBackend.Sync();
            TextEncoderBackend.FreeWeights(_gemma.EnumerateWeights());
            TextEncoderBackend.FreeWeights(_connectors.EnumerateWeights());
            // Host-materialize the four embeddings so they survive activation sweeps (a never-host-read tensor
            // loses its only copy in FreeActivations), then drop the TE's device activations + pool slack so
            // the resident-prefix sizing below sees the reclaimed VRAM. The transformer's RoPE table cache is
            // host-built, so it survives the sweep too.
            // LOAD-BEARING for TextEncoderDevice placement: these host reads ARE the cross-device boundary —
            // the denoiser's backend re-uploads the conditioning from the host copies.
            _ = encVideoPos.DataPointer; _ = encAudioPos.DataPointer;
            _ = encVideoNeg.DataPointer; _ = encAudioNeg.DataPointer;
            TextEncoderBackend.FreeActivations();
            TextEncoderBackend.TrimMemoryPool();
            _cachedVideoPos?.Dispose(); _cachedAudioPos?.Dispose();
            _cachedVideoNeg?.Dispose(); _cachedAudioNeg?.Dispose();
            _cachedVideoPos = encVideoPos; _cachedAudioPos = encAudioPos;
            _cachedVideoNeg = encVideoNeg; _cachedAudioNeg = encAudioNeg;
            _cachedPosKey = (int[])promptTokens.Clone();
            _cachedNegKey = (int[])negativeTokens.Clone();
            Logs.Info($"[ltx2-phase] TE(Gemma)+connectors+free: {phase.ElapsedMilliseconds} ms");
        }
        phase.Restart();

        // 2. Denoise both streams. The 22B fp8 DiT (~22 GB) doesn't fit resident alongside activations on 24 GB, so
        // stream its 48 blocks on/off device (the shared modulation tables plus a VRAM-sized, cross-generation
        // persistent block prefix stay resident). CPU/Vulkan (no streaming cache) preload everything eagerly.
        HartsyInference.Core.MemoryManagement.BlockStreamingController? streamer = null;
        int residentBlocks = 0;
        if (Backend.StreamingCache is not null)
        {
            Backend.PreloadWeights(_transformer.EnumerateSharedWeights());
            HartsyInference.Core.MemoryManagement.IStreamingBlock[] blocks =
                new HartsyInference.Core.MemoryManagement.IStreamingBlock[_transformer.BlockCount];
            for (int b = 0; b < blocks.Length; b++) blocks[b] = _transformer.GetBlock(b);

            // Resident prefix: park as many whole blocks as free VRAM allows and stream only the remainder.
            // Streaming a block costs its full weight bytes on EVERY forward (2 CFG forwards × steps), so each
            // resident block saves ~2·steps re-uploads. Headroom covers activations, the prefetch window, and
            // pool slack; tune via HARTSY_LTX2_HEADROOM_MB (0 disables residency). The count is sized ONCE and
            // pinned across generations; under KEEP_MODELS the prefix weights themselves also stay resident, so
            // the PreloadWeights below is a cache-hit no-op on every generation after the first.
            long blockBytes = blocks[0].EstimatedWeightBytes;
            // 3072 MB default headroom fits all 48 fp8 blocks (~18 GB) resident at 512×320 (freeing the connectors
            // above reclaimed the ~4 GB that used to force streaming) while leaving room for activations + the VAE
            // decode; the sizing auto-shrinks the resident count at larger geometries.
            long headroomMb = long.TryParse(Environment.GetEnvironmentVariable("HARTSY_LTX2_HEADROOM_MB"), out long hm) ? hm : 3072;
            long tokenLoad = (long)sv + audioFrames;
            IEnumerable<Tensor> BlockRangeWeights(int from, int to)
            {
                for (int b = from; b < to; b++)
                    foreach (Tensor t in blocks[b].EnumerateWeights()) yield return t;
            }
            if (_prefixResident && tokenLoad > _prefixSizedTokens)
            {
                // Bigger grid than the pinned sizing reserved headroom for — release and re-size below.
                Backend.FreeWeights(BlockRangeWeights(0, _residentPrefixBlocks));
                Backend.TrimMemoryPool();
                _prefixResident = false;
                _residentPrefixBlocks = -1;
                Logs.Info($"[ltx2-phase] resident prefix released for re-size (token load {tokenLoad} > sized {_prefixSizedTokens}).");
            }
            // Both sizings below read FreeMemoryBytes, which counts pool-retained blocks as USED. Without this trim
            // the figure still carries the previous generation's VAE-decode transients (~5 GB) and the Gemma
            // encode's, so the prefix gets sized against phantom pressure. That produced a two-generation
            // ping-pong through the SwarmUI API: a 48-block generation evicts for the decode, the next one
            // measures the un-trimmed pool, pins only ~22 blocks and streams the other 26 (~30 s slower), and
            // because it never fills VRAM it does not evict — so the cycle repeats. Pool slack is not spendable
            // on a 369 MB weight upload anyway, so trimming costs nothing real.
            if (!_prefixResident || _prefixTailEvicted) { Backend.TrimMemoryPool(); _prefixTailEvicted = false; }
            if (_residentPrefixBlocks < 0 || tokenLoad > _prefixSizedTokens)
            {
                // With the TE on its own device (TextEncoderBackend) the Gemma phase never touched this GPU, so
                // FreeMemoryBytes naturally reports the full budget here — no placement-specific handling needed.
                long spendable = Backend.FreeMemoryBytes() - headroomMb * 1024 * 1024;
                _residentPrefixBlocks = (int)Math.Clamp(spendable / Math.Max(blockBytes, 1), 0, blocks.Length);
                _prefixSizedTokens = tokenLoad;
            }
            residentBlocks = _residentPrefixBlocks;
            if (!_prefixResident && residentBlocks > 0)
            {
                // Re-upload path (after a TE-miss eviction): free VRAM can be tighter than the gen-1 sizing saw
                // (the video VAE + vocoder auto-promote resident on first decode). Squeeze THIS generation to
                // what fits but keep the pin — a later generation starts from the kept prefix plus trimmed pool
                // slack and tops the prefix back up to the pinned max (PreloadWeights is idempotent).
                long spendable = Backend.FreeMemoryBytes() - headroomMb * 1024 * 1024;
                int fit = (int)Math.Clamp(spendable / Math.Max(blockBytes, 1), 0, blocks.Length);
                if (fit < residentBlocks)
                {
                    Logs.Info($"[ltx2-phase] pinned prefix {residentBlocks} squeezed to {fit} for this generation " +
                        $"(free VRAM); pin kept — the next generation tops back up.");
                    residentBlocks = fit;
                }
            }
            if (residentBlocks > 0)
            {
                Backend.PreloadWeights(BlockRangeWeights(0, residentBlocks));
            }
            if (residentBlocks < blocks.Length)
            {
                HartsyInference.Core.MemoryManagement.IStreamingBlock[] streamed =
                    new HartsyInference.Core.MemoryManagement.IStreamingBlock[blocks.Length - residentBlocks];
                Array.Copy(blocks, residentBlocks, streamed, 0, streamed.Length);
                // prefetchAhead=2 double-buffers the streamed remainder (sources are pinned by the controller, so
                // the async H2D genuinely overlaps compute now).
                streamer = new HartsyInference.Core.MemoryManagement.BlockStreamingController(Backend.StreamingCache, streamed, prefetchAhead: 2, retainBehind: 0);
                int prefix = residentBlocks;
                HartsyInference.Core.MemoryManagement.BlockStreamingController s = streamer;
                _transformer.BeforeBlockForward = i => { if (i >= prefix) s.BeforeBlockForward(i - prefix); };
                streamer.Prime();
            }
            Logs.Info($"LTX-2 streaming: {blocks.Length} blocks × {blockBytes / (1024 * 1024)} MB, resident prefix {residentBlocks}{(_prefixResident ? " (persistent, no re-upload)" : "")}, streamed {blocks.Length - residentBlocks} ({(blocks.Length - residentBlocks) * blockBytes / (1024 * 1024)} MB/forward)");
        }
        else
        {
            Backend.PreloadWeights(_transformer.EnumerateWeights());
        }
        Tensor videoLat = SeedGenerator.CreateNoise(new TensorShape(sv, videoChannels), seed);
        Tensor audioLat = SeedGenerator.CreateNoise(new TensorShape(audioFrames, audioChannels), seed ^ 0x5D2B);
        // Distilled checkpoints baked their sigma schedule in, so it replaces the dynamic flow-match shift outright
        // and a different step count is not a valid schedule for them.
        float[] tsteps = _config.FixedSigmas ?? StretchTerminal(
            LancePipelineCommon.BuildShiftedTimesteps(steps, shift), _config, shift);
        Logs.Info($"[ltx2-phase] DiT preload+prime: {phase.ElapsedMilliseconds} ms");

        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float dt = tsteps[k] - tsteps[k + 1];
            float tEmb = tsteps[k] * _config.TimestepScaleMultiplier;   // flow sigma (≈1..0) scaled to ≈0..1000

            Tensor vCondV, vCondA, vUncondV, vUncondA;
            bool paired = !unguided;
            if (unguided)
            {
                // At guidance 1 the pair reduces to the conditional branch, so running the unconditional one is pure
                // waste — half the DiT work per step, which is most of the step. The distilled schedule always
                // lands here.
                (vCondV, vCondA) = _transformer.Forward(Backend, videoLat, audioLat, encVideoPos, encAudioPos,
                    tEmb, (tLat, hLat, wLat), audioFrames, frameRate, null, null);
                vUncondV = vCondV;
                vUncondA = vCondA;
            }
            else
            {
                // CFG-paired forward: both branches share each block's (streamed) weights within the step — half the
                // weight traffic of two sequential forwards.
                ((vCondV, vCondA), (vUncondV, vUncondA)) = _transformer.ForwardCfgPair(
                    Backend, videoLat, audioLat, encVideoPos, encAudioPos, encVideoNeg, encAudioNeg,
                    tEmb, (tLat, hLat, wLat), audioFrames, frameRate);
            }

            // Device CFG+Euler, in-place on the resident latents: z += (g·cond + (1−g)·uncond)·(−dt) ≡ z −= v·dt.
            // The latents stay GPU-resident across the whole loop; the final host read (UnpackVideoLatents) syncs.
            // With guidance 1 the cond tensor is passed for both operands, which the op collapses to plain Euler.
            Backend.CfgEulerStep(videoLat, vCondV, vUncondV, guidance, -dt);
            AudioCfgEulerStep(audioLat, vCondA, vUncondA, audioGuidance, audioRescale, -dt);
            // On the step-graph path the four velocities are transformer-owned fixed buffers (rewritten next step) —
            // don't dispose them; on the eager path they're fresh and must be freed. The unguided path aliases the
            // cond tensors into the uncond slots, so it must not double-free them.
            if (!_transformer.StepGraphActive)
            {
                vCondV.Dispose();
                vCondA.Dispose();
                if (paired) { vUncondV.Dispose(); vUncondA.Dispose(); }
            }

            Backend.Sync();
            sw.Stop();
            Logs.Info($"[ltx2-phase] step {k + 1}/{steps}: {sw.ElapsedMilliseconds} ms (paired CFG)");
            // The preview drain reads the latent on host, which EVICTS it from the GPU — incompatible with the step
            // graph, whose capture bakes the resident latent's device address (a re-upload would move it). Skip the
            // preview while the graph is active (correctness > the optional live thumbnail).
            if (onProgress is not null && !_transformer.StepGraphActive)
            {
                Tensor preview = ExtractMiddleFrame(videoLat, tLat, hLat, wLat, videoChannels);
                onProgress.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
                {
                    Latent = preview,
                    LatentArch = LatentArchitecture.Ltx,
                });
                preview.Dispose();
            }
        }

        Backend.Sync();
        // The captured graph bakes the DiT weight pointers; invalidate before any FreeWeights below so the next
        // generation re-warms and re-captures instead of replaying against freed memory.
        _transformer.InvalidateStepGraph(Backend);
        phase.Restart();
        _transformer.BeforeBlockForward = null;
        if (streamer is not null) { streamer.EvictAll(); streamer.Dispose(); }
        if (KeepModelsResident && Backend.StreamingCache is not null && residentBlocks > 0)
        {
            // KEEP_MODELS: shared weights + the block prefix stay device-resident for the next generation
            // (its PreloadWeights becomes a cache-hit no-op and the streaming boundary stays pinned). The
            // streamed suffix is already evicted — FreeWeights it anyway to reclaim any lingering cached
            // dtype casts — then trim so the VAE decode gets the streaming window's VRAM.
            int blockCount = _transformer.BlockCount;
            int keptPrefix = residentBlocks;
            IEnumerable<Tensor> SuffixWeights()
            {
                for (int b = keptPrefix; b < blockCount; b++)
                    foreach (Tensor t in _transformer.GetBlock(b).EnumerateWeights()) yield return t;
            }
            Backend.FreeWeights(SuffixWeights());
            Backend.TrimMemoryPool();
            _prefixResident = true;
            Logs.Info($"[ltx2-phase] resident prefix kept across generations: {residentBlocks} blocks " +
                $"(KEEP_MODELS; free {Backend.FreeMemoryBytes() >> 20} MB for decode)");
        }
        else
        {
            // Frees shared weights + the resident prefix (streamed blocks are already evicted — FreeWeights
            // skips non-cached tensors). The VAE decode needs this VRAM back.
            Backend.FreeWeights(_transformer.EnumerateWeights());
            _prefixResident = false;
        }
        // The persistent prefix must never starve the decode: the GPU-ported video VAE peaks at a handful of
        // full-output-grid stages plus conv workspace — if free VRAM is short of that, drop the prefix for
        // this generation (an OOM is worse than one re-upload). A VAE on its OWN device (VaeBackend) never
        // contends with the prefix, so the evict is skipped when split.
        //
        // Per output pixel, scaled by the decode's activation width — the peak is a fixed COUNT of full-grid
        // tensors, so it is linear in the element size (F32 keeps the measured 160 B/px; BF16 halves it).
        // MEASURED, not derived: a BF16 decode at 768x512x97f OOMs with 2486 MB free, so the reserve must stay
        // well above the ~2.5 GB a naive count of the stage tensors suggests — conv workspace and pool
        // fragmentation are real. At this geometry BF16 still evicts the prefix; the saving it buys is decode
        // TIME (3.9 s -> 2.8 s) and headroom at smaller geometries, not the eviction.
        // Not simply halved for BF16: the fixed conv workspace and pool fragmentation do not scale with the
        // element width. Bracketed by experiment at 768x512x97f — BF16 OOMs outright at 2486 MB free and only
        // squeaks through on allocator retries at ~3.05 GB, so 104 B/px (~4.0 GB here) is the first reserve with
        // real margin.
        long perPixel = _vae.ComputeDtype == DType.F32 ? 160L : 104L;
        long decodeNeed = Math.Max(3L << 30, (long)numFrames * height * width * perPixel);
        // The diffusion decoder's stage-5 trunk is a transformer over patchified pixels, and nothing here models its
        // windowed-attention workspaces — a bracket that is merely too small silently skips eviction and OOMs mid
        // decode. Give it everything the prefix is holding instead and pay the re-preload.
        if (_diffusionVae is not null && _prefixResident && ReferenceEquals(VaeBackend, Backend))
        {
            Logs.Info($"[ltx2-phase] dropping the resident prefix for the diffusion VAE decode "
                + $"(free {Backend.FreeMemoryBytes() >> 20} MB before).");
            Backend.FreeWeights(_transformer.EnumerateWeights());
            long freeAfterDrop = Backend.FreeMemoryBytes();
            Backend.TrimMemoryPool();
            Logs.Info($"[ltx2-phase] diffusion-VAE prefix drop: free {freeAfterDrop >> 20} MB after FreeWeights, "
                + $"{Backend.FreeMemoryBytes() >> 20} MB after TrimMemoryPool.");
            _prefixResident = false;
        }
        else if (_prefixResident && ReferenceEquals(VaeBackend, Backend) && Backend.FreeMemoryBytes() < decodeNeed)
        {
            // Free only the TAIL of the prefix the decode is actually short of, from the end so the pin at
            // _residentPrefixBlocks still describes the survivors. Dropping all 48 blocks to reclaim a ~400 MB
            // deficit cost a full 21.5 GB re-preload on the next generation (3.5 s) to buy one block's worth.
            long blockBytes = Math.Max(_transformer.GetBlock(0).EstimatedWeightBytes, 1);
            // One block of MARGIN on top of the deficit. decodeNeed's 104 B/px is a bracket, and the full-prefix
            // eviction used to hide its error under ~21 GB of slack: freeing exactly the deficit reproduced
            // "OOM on async first attempt (free 42.9 MB)" allocator retries mid-decode that this workload had
            // never logged before. A block is the natural granularity and costs ~90 ms of re-preload.
            long deficit = decodeNeed - Backend.FreeMemoryBytes();
            int toFree = (int)Math.Min(residentBlocks, (deficit + blockBytes - 1) / blockBytes + 1);
            int keep = residentBlocks - toFree;
            IEnumerable<Tensor> EvictedTail()
            {
                for (int b = keep; b < residentBlocks; b++)
                    foreach (Tensor t in _transformer.GetBlock(b).EnumerateWeights()) yield return t;
            }
            Logs.Info($"[ltx2-phase] evicting {toFree} of {residentBlocks} resident blocks for VAE decode " +
                $"(free {Backend.FreeMemoryBytes() >> 20} MB < ~{decodeNeed >> 20} MB estimated decode peak).");
            Backend.FreeWeights(EvictedTail());
            Backend.TrimMemoryPool();
            // The estimate is a bracket, not an allocator model, so verify rather than assume: if the tail was not
            // enough, drop the whole prefix — an OOM in the decode is worse than the re-upload this exists to avoid.
            if (Backend.FreeMemoryBytes() < decodeNeed)
            {
                Logs.Info($"[ltx2-phase] tail eviction left only {Backend.FreeMemoryBytes() >> 20} MB — dropping the whole prefix.");
                Backend.FreeWeights(_transformer.EnumerateWeights());
                Backend.TrimMemoryPool();
                _prefixResident = false;
            }
            else _prefixTailEvicted = true;
        }
        // The four text embeddings are NOT disposed — they are the cross-generation prompt cache (tiny,
        // host-materialized; freed on the next cache miss or by the finalizer drain).

        // 3. Decode video. UnpackVideoLatents is a host loop — LOAD-BEARING for VaeDevice placement: it IS the
        // cross-device boundary, so the (possibly separate) VAE backend uploads from the host copy.
        Tensor videoVaeLatent = UnpackVideoLatents(videoLat, tLat, hLat, wLat, videoChannels);
        videoLat.Dispose();
        Logs.Info($"[ltx2-phase] latent unpack (host): {phase.ElapsedMilliseconds} ms");
        phase.Restart();
        ProbeTensor("pre-decode videoVaeLatent", videoVaeLatent);
        Tensor rgb = DecodeVideo(videoVaeLatent, tLat, hLat, wLat, seed);
        videoVaeLatent.Dispose();
        if (!ReferenceEquals(VaeBackend, Backend)) VaeBackend.Sync();
        Backend.Sync();
        Logs.Info($"[ltx2-phase] video VAE decode: {phase.ElapsedMilliseconds} ms");
        phase.Restart();
        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = VideoRgbFrames.ExtractFrame(rgb, i);
        rgb.Dispose();
        Logs.Info($"[ltx2-phase] rgb frame extract: {phase.ElapsedMilliseconds} ms");
        phase.Restart();

        // 4. Decode audio (optional — requires the audio VAE + vocoder).
        float[][]? audio = null;
        int audioSampleRate = 0;
        if (_audioVae is not null && _vocoder is not null)
            audio = DecodeAudio(audioLat, audioFrames, out audioSampleRate);
        audioLat.Dispose();
        Logs.Info($"[ltx2-phase] audio decode total: {phase.ElapsedMilliseconds} ms");

        Logs.Info($"LTX-2 complete ({frames.Length} frames" + (audio is not null ? " + audio" : "") + $", seed={seed})");
        return new Ltx2Result(frames, width, height, seed, audio, audioSampleRate);
    }

    /// <summary>Runs Gemma over the (register-padded) tokens, relayouts the 49 hidden states into the connector's
    /// <c>channel·49+layer</c> feature layout, and returns the per-modality text embeddings (video <c>[seq,4096]</c>,
    /// audio <c>[seq,2048]</c>). Caller owns both tensors.</summary>
    private (Tensor Video, Tensor Audio) EncodeText(int[] tokens)
    {
        int real = tokens.Length;
        int seq = ((real + ConnectorRegisters - 1) / ConnectorRegisters) * ConnectorRegisters;
        if (seq == 0) seq = ConnectorRegisters;
        // Some families condition at a fixed length regardless of prompt (Gemma 4 at 1024). Length is part of the
        // conditioning because the connector replaces learnable registers positionally, so it is padded UP to that
        // here — where validMask still marks only the real tokens. Padding upstream in the tokenizer instead would
        // make `real` count the pad tokens and present them to the connector as content.
        if (seq < MinimumTextConditioningLength)
            seq = ((MinimumTextConditioningLength + ConnectorRegisters - 1) / ConnectorRegisters) * ConnectorRegisters;

        // Right-pad to a register multiple. The Gemma encoder applies only a causal mask (no padding mask), so
        // padding on the right keeps real tokens (at the front) from attending to pad tokens; validMask marks them.
        int[] padded = new int[seq];
        for (int i = 0; i < real; i++) padded[i] = tokens[i];
        float[] validMask = new float[seq];
        for (int i = 0; i < real; i++) validMask[i] = 1f;

        int[] layerIndices = new int[GemmaLayers];
        for (int i = 0; i < GemmaLayers; i++) layerIndices[i] = i;       // 0=embeddings, 1..48=post-layer
        Stopwatch sub = Stopwatch.StartNew();
        Tensor multi = _gemma.EncodeMultiLayer(TextEncoderBackend, [padded], layerIndices);  // [1, seq, 49·3840] layer-outer
        Logs.Info($"[ltx2-phase]   gemma encode ({seq} tok): {sub.ElapsedMilliseconds} ms");
        sub.Restart();

        // Relayout to channel-outer (feature = channel·49 + layer), which the connector consumes.
        Tensor feats = new Tensor(new TensorShape(seq, GemmaLayers * GemmaCaptionChannels), DType.F32);
        float* sp = (float*)multi.DataPointer;
        float* dp = (float*)feats.DataPointer;
        long stride = (long)GemmaLayers * GemmaCaptionChannels;
        for (int t = 0; t < seq; t++)
        {
            float* srcRow = sp + (long)t * stride;     // [layer·3840 + channel]
            float* dstRow = dp + (long)t * stride;     // [channel·49 + layer]
            for (int l = 0; l < GemmaLayers; l++)
                for (int c = 0; c < GemmaCaptionChannels; c++)
                    dstRow[(long)c * GemmaLayers + l] = srcRow[(long)l * GemmaCaptionChannels + c];
        }
        multi.Dispose();
        Logs.Info($"[ltx2-phase]   gemma relayout (host): {sub.ElapsedMilliseconds} ms");
        sub.Restart();

        (Tensor video, Tensor audio) = _connectors.Forward(TextEncoderBackend, feats, validMask);
        feats.Dispose();
        Logs.Info($"[ltx2-phase]   connectors: {sub.ElapsedMilliseconds} ms");
        return (video, audio);
    }

    /// <summary>CFG + Euler for the audio stream, optionally rescaling the guided velocity toward the conditional's
    /// std (diffusers' <c>rescale_noise_cfg</c>). Std only — the reference has no mean-matching term, and adding one
    /// injects a DC offset into the latent. The blend is affine in the guided velocity, so only the scalars come to
    /// the host and the step stays on device (the latent is never written host-side, which would go stale against
    /// its GPU cache).</summary>
    private void AudioCfgEulerStep(Tensor z, Tensor cond, Tensor uncond, float guidance, float rescale, float delta)
    {
        if (rescale <= 0f)
        {
            Backend.CfgEulerStep(z, cond, uncond, guidance, delta);
            return;
        }
        Backend.Sync();
        long n = cond.ElementCount;
        float* c = (float*)cond.DataPointer;
        float* u = (float*)uncond.DataPointer;
        double sumC = 0, sumG = 0;
        for (long i = 0; i < n; i++)
        {
            sumC += c[i];
            sumG += guidance * c[i] + (1f - guidance) * u[i];
        }
        double meanC = sumC / n, meanG = sumG / n;
        double varC = 0, varG = 0;
        for (long i = 0; i < n; i++)
        {
            double dc = c[i] - meanC;
            double dg = (guidance * c[i] + (1f - guidance) * u[i]) - meanG;
            varC += dc * dc;
            varG += dg * dg;
        }
        double factor = Math.Sqrt(varC / n) / Math.Max(Math.Sqrt(varG / n), 1e-8);
        double a = rescale * factor + (1f - rescale);
        Backend.CfgEulerStep(z, cond, uncond, guidance, (float)(a * delta));
    }

    /// <summary>Test seam for <see cref="StretchTerminal"/> — the schedule is pure math and worth pinning against
    /// ComfyUI's numbers without standing up a pipeline.</summary>
    internal static float[] StretchTerminalForTests(float[] sigmas, LtxVideo2Config config, float shift)
        => StretchTerminal(sigmas, config, shift);

    /// <summary>Rescales a shifted flow-match schedule so its last non-zero sigma lands on the config's terminal
    /// value, matching LTX's own scheduler (<c>stretch=True, terminal=0.1</c> in the shipped templates). Without it
    /// the denoise stops early: the shift grows with token count, so at 1280x736x97f the schedule ends at sigma
    /// 0.817 and drops to zero in one step. <c>HARTSY_LTX2_SIGMA_STRETCH=0</c> restores the un-stretched schedule.</summary>
    private static float[] StretchTerminal(float[] sigmas, LtxVideo2Config config, float shift)
    {
        if (!config.SigmaStretch || Environment.GetEnvironmentVariable("HARTSY_LTX2_SIGMA_STRETCH") == "0")
        {
            return sigmas;
        }
        // The schedule ends in an explicit 0; the value to pin is the last NON-zero entry.
        int last = sigmas.Length - 1;
        while (last > 0 && sigmas[last] == 0f) last--;
        if (last <= 0) return sigmas;
        float scale = (1f - sigmas[last]) / (1f - config.SigmaTerminal);
        if (scale <= 0f) return sigmas;
        for (int i = 0; i <= last; i++)
        {
            if (sigmas[i] != 0f) sigmas[i] = 1f - ((1f - sigmas[i]) / scale);
        }
        Logs.Debug($"[ltx2] sigma stretch: shift={shift:F3} last {sigmas[last]:F4} (terminal {config.SigmaTerminal:F2})");
        return sigmas;
    }

    /// <summary>Decodes the unpacked video latent to RGB through whichever decoder the checkpoint shipped. The
    /// conv decoder un-normalizes internally; the diffusion decoder does not, and needs injected noise — seeded off
    /// the generation seed so a repeated generation still reproduces.</summary>
    private Tensor DecodeVideo(Tensor videoVaeLatent, int tLat, int hLat, int wLat, int seed)
    {
        if (_diffusionVae is null)
        {
            return _vae.Decode(VaeBackend, videoVaeLatent);
        }
        Tensor denorm = LtxVideo2VaeDecoder.DenormalizeLatent(videoVaeLatent, _videoLatentsMean, _videoLatentsStd);
        ProbeTensor("diffusion-decoder latent (denormalized)", denorm);
        Tensor noise = SeedGenerator.CreateNoise(_diffusionVae.NoiseShape(tLat, hLat, wLat), seed ^ 0x2D5B);
        try
        {
            return _diffusionVae.Decode(VaeBackend, denorm, noise);
        }
        finally
        {
            denorm.Dispose();
            noise.Dispose();
        }
    }

    /// <summary>Writes a stage tensor as raw little-endian F32 + a shape sidecar into
    /// <c>HARTSY_LTX2_AUDIO_DUMP</c>; no-op when unset. Lets the reference implementation decode OUR tensors.</summary>
    private static void DumpTensor(string name, Tensor tensor)
    {
        if (Environment.GetEnvironmentVariable("HARTSY_LTX2_AUDIO_DUMP") is not { Length: > 0 } dir)
        {
            return;
        }
        Directory.CreateDirectory(dir);
        Tensor f32 = tensor.DType == DType.F32 ? tensor : tensor.CastTo(DType.F32);
        long n = f32.ElementCount;
        using (FileStream fs = File.Create(Path.Combine(dir, name + ".bin")))
        {
            fs.Write(new ReadOnlySpan<byte>((byte*)f32.DataPointer, checked((int)(n * 4))));
        }
        File.WriteAllText(Path.Combine(dir, name + ".json"),
            "{\"shape\":[" + string.Join(",", Enumerable.Range(0, f32.Shape.Rank).Select(i => f32.Shape[i])) + "]}");
        if (!ReferenceEquals(f32, tensor)) f32.Dispose();
        Logs.Warning($"[ltx2-dump] wrote {name} {tensor.Shape} to {dir}");
    }

    /// <summary>Logs min/max/mean/rms for a stage output under <c>HARTSY_LTX2_PROBE=1</c>; no-op otherwise.</summary>
    private static void ProbeTensor(string label, Tensor tensor)
    {
        if (Environment.GetEnvironmentVariable("HARTSY_LTX2_PROBE") != "1")
        {
            return;
        }
        Tensor f32 = tensor.DType == DType.F32 ? tensor : tensor.CastTo(DType.F32);
        float* p = (float*)f32.DataPointer;
        long n = f32.ElementCount;
        float mn = float.MaxValue, mx = float.MinValue;
        double sum = 0, sumSq = 0;
        long nanCount = 0, infCount = 0;
        for (long e = 0; e < n; e++)
        {
            float v = p[e];
            if (float.IsNaN(v)) { nanCount++; continue; }
            if (float.IsInfinity(v)) { infCount++; continue; }
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
            sumSq += (double)v * v;
        }
        double rms = n > 0 ? Math.Sqrt(sumSq / n) : 0d;
        Logs.Warning($"[ltx2-probe] {label}: min={mn:F5} max={mx:F5} mean={sum / n:F5} rms={rms:F5} "
            + $"nan={nanCount} inf={infCount} n={n} shape={tensor.Shape}");
        if (!ReferenceEquals(f32, tensor)) f32.Dispose();
    }

    /// <summary>Audio VAE + vocoder: denormalize → unpack <c>[L,128]→[1,8,L,16]</c> → mel → 48 kHz waveform.</summary>
    private float[][] DecodeAudio(Tensor audioLat, int audioFrames, out int sampleRate)
    {
        Stopwatch phase = Stopwatch.StartNew();
        // Audio VAE + vocoder follow VaeDevice: both handoffs (UnpackAudioLatents in, the PCM copy-out below)
        // are host loops already, and leaving audio decode on the primary would reintroduce exactly the
        // decode-phase VRAM pressure the VaeDevice split removes.
        VaeBackend.PreloadWeights(_audioVae!.EnumerateWeights());

        int latentChannels = 8;
        int melLat = _config.AudioInChannels / latentChannels;   // 128 / 8 = 16
        ProbeTensor("audio latent (raw, pre-denorm)", audioLat);
        DumpTensor("audio_latent_raw", audioLat);
        Tensor unpacked = UnpackAudioLatents(audioLat, audioFrames, latentChannels, melLat);
        ProbeTensor("audio latent (unpacked, post-denorm)", unpacked);
        Tensor mel = _audioVae.Decode(VaeBackend, unpacked);     // [1, 2, T, 64]
        ProbeTensor("audio VAE out (log-mel)", mel);
        DumpTensor("audio_mel", mel);
        unpacked.Dispose();
        VaeBackend.Sync();
        Logs.Info($"[ltx2-phase]   audio VAE (preload+unpack+decode): {phase.ElapsedMilliseconds} ms");
        phase.Restart();
        // The vocoder manages its own weights (no bulk EnumerateWeights); ops fault them in on demand.
        Tensor wave = _vocoder!.Forward(VaeBackend, mel);        // [1, channels, samples]
        ProbeTensor("vocoder out (waveform)", wave);
        mel.Dispose();
        VaeBackend.Sync();
        Logs.Info($"[ltx2-phase]   vocoder: {phase.ElapsedMilliseconds} ms");
        VaeBackend.FreeWeights(_audioVae.EnumerateWeights());

        int channels = (int)wave.Shape[1], samples = (int)wave.Shape[2];
        float[][] pcm = new float[channels][];
        float* wp = (float*)wave.DataPointer;
        for (int c = 0; c < channels; c++)
        {
            pcm[c] = new float[samples];
            for (int s = 0; s < samples; s++) pcm[c][s] = wp[(long)c * samples + s];
        }
        wave.Dispose();
        sampleRate = _vocoder.SampleRate;   // 48 kHz (BWE) or 24 kHz (single-stage), set at LoadWeights
        return pcm;
    }

    /// <summary>Denormalizes (per-channel stats) and unpacks audio tokens <c>[L, C·M]</c> (channel·M+mel) →
    /// <c>[1, C, L, M]</c>.</summary>
    private Tensor UnpackAudioLatents(Tensor tokens, int frames, int channels, int mel)
    {
        Tensor outT = new Tensor(new TensorShape([1L, channels, frames, mel]), DType.F32);
        float* sp = (float*)tokens.DataPointer;
        float* dp = (float*)outT.DataPointer;
        bool denorm = _audioLatentsMean is not null && _audioLatentsStd is not null;
        // The latent stats are stored over the packed feature axis (channel·mel = 128). Index by the packed feature
        // when the stat length matches that; fall back to per-channel if a [channels]-length stat is supplied.
        bool perFeature = denorm && _audioLatentsMean!.Length == channels * mel;
        long frameStride = (long)mel;
        for (int fI = 0; fI < frames; fI++)
            for (int c = 0; c < channels; c++)
                for (int mI = 0; mI < mel; mI++)
                {
                    float v = sp[(long)fI * channels * mel + (long)c * mel + mI];
                    if (denorm)
                    {
                        int si = perFeature ? c * mel + mI : c;
                        v = v * _audioLatentsStd![si] + _audioLatentsMean![si];
                    }
                    dp[(((long)c * frames + fI)) * frameStride + mI] = v;
                }
        return outT;
    }

    /// <summary>Unpacks video tokens <c>[S, C]</c> (f,h,w order, channel-last) → <c>[1, C, T, H, W]</c>.</summary>
    private static Tensor UnpackVideoLatents(Tensor tokens, int t, int h, int w, int channels)
    {
        Tensor outT = new Tensor(new TensorShape([1L, channels, t, h, w]), DType.F32);
        float* sp = (float*)tokens.DataPointer;
        float* dp = (float*)outT.DataPointer;
        long spatial = (long)t * h * w;
        for (int ti = 0; ti < t; ti++)
            for (int hi = 0; hi < h; hi++)
                for (int wi = 0; wi < w; wi++)
                {
                    long token = ((long)ti * h + hi) * w + wi;
                    for (int c = 0; c < channels; c++)
                        dp[(long)c * spatial + token] = sp[token * channels + c];
                }
        return outT;
    }

    /// <summary>Drops the cross-generation caches: the persistent resident-prefix device weights and the four
    /// host-materialized prompt embeddings. Model switches dispose the pipeline while the backend lives on, and
    /// the weight cache keeps preloaded tensors (and their device copies) alive — free eagerly so a non-empty
    /// pipeline cache can't strand the prefix VRAM (the Chroma→Krea2 eviction lesson).</summary>
    protected override void DisposeCore()
    {
        try
        {
            if (_prefixResident)
            {
                Backend.FreeWeights(_transformer.EnumerateWeights());
                Backend.TrimMemoryPool();
                _prefixResident = false;
            }
        }
        catch (Exception ex)
        {
            Logs.Error($"LTX-2 pipeline dispose: failed to free the resident prefix: {ex}");
        }
        _cachedVideoPos?.Dispose(); _cachedAudioPos?.Dispose();
        _cachedVideoNeg?.Dispose(); _cachedAudioNeg?.Dispose();
        _cachedVideoPos = _cachedAudioPos = _cachedVideoNeg = _cachedAudioNeg = null;
        _cachedPosKey = null; _cachedNegKey = null;
    }

    /// <summary>Middle latent frame <c>[1, C, H, W]</c> for latent2rgb previews.</summary>
    private static Tensor ExtractMiddleFrame(Tensor tokens, int t, int h, int w, int channels)
    {
        Tensor outT = new Tensor(new TensorShape([1L, channels, h, w]), DType.F32);
        float* sp = (float*)tokens.DataPointer;
        float* dp = (float*)outT.DataPointer;
        long frameBase = (long)(t / 2) * h * w;
        for (int hi = 0; hi < h; hi++)
            for (int wi = 0; wi < w; wi++)
            {
                long token = frameBase + (long)hi * w + wi;
                long pix = (long)hi * w + wi;
                for (int c = 0; c < channels; c++)
                    dp[(long)c * h * w + pix] = sp[token * channels + c];
            }
        return outT;
    }
}
