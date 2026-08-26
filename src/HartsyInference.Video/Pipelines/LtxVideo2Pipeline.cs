using HartsyInference.Core.Configuration;
using System.Diagnostics;
using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;
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
    private const int ConnectorRegisters = 128;     // text seq is padded to a multiple of this

    /// <summary>Shortest conditioning sequence to pad to, whatever the prompt length; 0 uses only the register
    /// multiple. LTX-2.5's Gemma 4 conditions at 1024; LTX-2.3 leaves this 0 and is unaffected.</summary>
    public int MinimumTextConditioningLength { get; init; }

    /// <summary>Runs the shipped LTX-2.5 two-stage flow: denoise at HALF resolution, run the learned x2 latent
    /// upsampler, then refine 3 steps at full resolution. Requires <see cref="LatentUpsampler"/>; ignored without
    /// it. The half-resolution stage keeps the token count inside the 1024–4096 window the shift fit was
    /// calibrated on, which is the whole point — a full-resolution single pass never was.</summary>
    public bool TwoStage { get; init; }

    /// <summary>The learned x2 spatial latent upsampler for <see cref="TwoStage"/>. Weights live on the host and
    /// are uploaded/freed around the single stage-transition call.</summary>
    public LtxLatentUpsampler? LatentUpsampler { get; init; }

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
    private bool KeepModelsResident => VramLevers.KeepResident(Backend);
    private readonly ResidentPrefixPin _prefixPin = new ResidentPrefixPin();
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
        // Sampler selection is NOT wired on this family (2026-08-20 audit): its denoise step is a host-side
        // Euler in a different algebraic form than IBackend.CfgEulerStep, its schedule is a raw float[] rather
        // than a FlowMatchEulerDiscreteScheduler, and one model evaluation drives TWO latents (video + audio) with separate guidance scales, which ISampler.Step cannot express. Converting it would change
        // default-Euler output, so it was deliberately skipped. Refuse a named sampler rather than accepting the
        // request and silently sampling with something else.
        if (!string.IsNullOrWhiteSpace(request.Scheduler))
        {
            throw new NotSupportedException(
                $"Sampler/schedule '{request.Scheduler}' is not available on LTX-2. This family has not been "
                + "converted to the sampler seam yet, so it samples with its own Euler step only. Leave the "
                + "sampler unset.");
        }


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
        // Two-stage geometry: stage 1 is the requested size halved and snapped DOWN to the latent grid, and the
        // upsampler doubles that — so a half-size that is not itself a whole number of latent cells loses one, and
        // a 1280x736 request renders 1280x704. The output dimensions are corrected here rather than at the end so
        // every downstream consumer (the result record included) sees what was actually rendered.
        bool twoStage = TwoStage && LatentUpsampler is not null;
        int hLatStage1 = 0, wLatStage1 = 0;
        if (twoStage)
        {
            // Both normalization helpers degrade to a plain COPY when a stat is absent, so a checkpoint without
            // per-channel statistics would silently feed the upsampler normalized latents and produce a plausible
            // wrong video. Two-stage without the stats is not a degraded mode.
            if (_videoLatentsMean is null || _videoLatentsStd is null)
                throw new InvalidOperationException("LTX-2.5 two-stage needs the video VAE's per-channel latent statistics: the latent upsampler is defined on UN-normalized latents.");
            (hLatStage1, wLatStage1, hLat, wLat) = TwoStageGrid(width, height, sp);
            height = hLat * sp;
            width = wLat * sp;
        }
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
                string refineNote = TwoStage && LatentUpsampler is not null
                    ? $" (plus the fixed 3-step refine stage — {distilledSteps}+3 total)" : "";
                Logs.Warning($"LTX-2 distilled: ignoring the requested {steps} steps — this checkpoint was distilled " +
                    $"onto a fixed {distilledSteps}-step schedule{refineNote}, and any other count is a different schedule.");
                steps = distilledSteps;
            }
        }
        float guidance = request.CfgScale ?? _config.GuidanceScale;
        // The reference carries a separate audio CFG scale; unset follows the video scale.
        float audioGuidance = _config.AudioGuidanceScale ?? guidance;
        // Both scales must be 1 to skip the unconditional branch: the single-branch path has no unconditional
        // velocity to give the audio Euler step, so a guided audio stream still needs the pair.
        bool unguided = guidance == 1f && audioGuidance == 1f;
        float audioRescale = _config.AudioGuidanceRescale;

        // Stage 1 is what the sampler actually denoises, so it owns the schedule's token count.
        int svStage1 = twoStage ? tLat * hLatStage1 * wLatStage1 : sv;
        float[]? refineSigmas = twoStage ? LtxVideo2Config.Ltx25TwoStageRefineSigmas : null;
        int refineSteps = refineSigmas is null ? 0 : refineSigmas.Length - 1;
        float shift = ComputeShift(svStage1, _config), formulaShift = FormulaShift(svStage1);
        // The shipped template runs euler_ancestral in both stages, but measured here it costs audio: same prompt,
        // seed and geometry, the 2-4 kHz dynamic range is 39.5 dB ancestral against 47.8 dB plain, and the noise
        // floor -5.6 dB against -14.2 dB. Three ancestral injections is enough to leave a broadband bed the 3-step
        // refine cannot re-absorb. Plain Euler is the default in both arms; HARTSY_LTX2_ANCESTRAL=1 opts back in.
        bool ancestral = EngineKnobs.Ltx2Ancestral.Value ?? _config.EulerAncestral;

        Logs.Info($"LTX-2 T2V+A: {numFrames}f {width}x{height}, {steps}{(twoStage ? $"+{refineSteps}" : "")} steps, " +
            $"cfg={guidance}, seed={seed} (video {tLat}x{(twoStage ? hLatStage1 : hLat)}x{(twoStage ? wLatStage1 : wLat)}" +
            $"={svStage1} tokens, audio {audioFrames} tokens, " +
            $"shift={shift:F3}{(shift == formulaShift ? "" : $", overriding the fit's {formulaShift:F3}")}, " +
            $"{(ancestral ? "euler_ancestral" : "euler")})");

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
            if (_prefixPin.Resident && ReferenceEquals(TextEncoderBackend, Backend))
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
                    _prefixPin.Resident = false;
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
        // 3072 MB default headroom fits all 48 fp8 blocks (~18 GB) resident at 512×320 while leaving room for
        // activations + the VAE decode; the sizing auto-shrinks the resident count at larger geometries.
        // PerStepTrim is off here alone: this denoise loop replays a captured step graph, and a per-step
        // compute-stream sync inside a graph replay is unverified on this model.
        BlockStreamingScope stream = BlockStreamingScope.Open(new BlockStreamingOptions
        {
            Backend = Backend,
            Denoiser = _transformer,
            ModelName = "LTX-2",
            HeadroomBytes = EngineKnobs.Ltx2HeadroomMb.Value * 1024 * 1024,
            TokenLoad = (long)sv + audioFrames,
            Pin = _prefixPin,
            PerStepTrim = false,
        });
        int residentBlocks = stream.ResidentPrefixBlocks;
        // Releases / re-uploads the DiT weights THIS generation parked on device, for the two-stage transition: the
        // ~2 GB F32 upsampler may have to displace them to fit. Deliberately excludes the streamed suffix — the
        // streaming controller owns those uploads and freeing them behind its back would leave it believing blocks
        // it no longer holds are resident.
        Action? releaseDitWeights = null, restoreDitWeights = null;
        if (residentBlocks > 0)
        {
            releaseDitWeights = () => Backend.FreeWeights(stream.EnumerateResidentWeights());
            restoreDitWeights = () => Backend.PreloadWeights(stream.EnumerateResidentWeights());
        }
        else if (!stream.Streaming)
        {
            releaseDitWeights = () => Backend.FreeWeights(_transformer.EnumerateWeights());
            restoreDitWeights = () => Backend.PreloadWeights(_transformer.EnumerateWeights());
        }
        Tensor videoLat = SeedGenerator.CreateNoise(new TensorShape(twoStage ? svStage1 : sv, videoChannels), seed);
        Tensor audioLat = SeedGenerator.CreateNoise(new TensorShape(audioFrames, audioChannels), seed ^ 0x5D2B);
        // Distilled checkpoints baked their sigma schedule in, so it replaces the dynamic flow-match shift outright
        // and a different step count is not a valid schedule for them.
        float[] tsteps = _config.FixedSigmas ?? StretchTerminal(
            LancePipelineCommon.BuildShiftedTimesteps(steps, shift), _config, shift);
        Logs.Info($"[ltx2-phase] DiT preload+prime: {phase.ElapsedMilliseconds} ms");

        DenoiseStage(videoLat, audioLat, tsteps, twoStage ? (tLat, hLatStage1, wLatStage1) : (tLat, hLat, wLat),
            audioFrames, frameRate, encVideoPos, encAudioPos, encVideoNeg, encAudioNeg,
            guidance, audioGuidance, audioRescale, unguided, 0, steps + refineSteps, videoChannels, onProgress,
            twoStage ? "s1 " : "", ancestral, seed);

        if (twoStage)
        {
            Backend.Sync();
            // The refine stage runs at a DIFFERENT grid and rebuilds the RoPE tables, which would strand the
            // pointers a stage-1 capture baked. Invalidate before the transition; the stage itself is then hidden
            // from the graph state machine entirely, so its grid never reaches the signature.
            _transformer.InvalidateStepGraph(Backend);
            (Tensor refineVideo, Tensor refineAudio) = UpsampleAndRenoise(videoLat, audioLat,
                tLat, hLatStage1, wLatStage1, videoChannels, refineSigmas![0], seed, releaseDitWeights, restoreDitWeights);
            videoLat.Dispose();
            audioLat.Dispose();
            videoLat = refineVideo;
            audioLat = refineAudio;
            Logs.Info($"[ltx2-two-stage] refine at {tLat}x{hLat}x{wLat}={sv} tokens ({width}x{height}), "
                + $"sigmas [{string.Join(", ", refineSigmas.Select(s => s.ToString("0.######", CultureInfo.InvariantCulture)))}], eager (no step graph).");
            _transformer.StepGraphSuspended = true;
            try
            {
                DenoiseStage(videoLat, audioLat, refineSigmas, (tLat, hLat, wLat), audioFrames, frameRate,
                    encVideoPos, encAudioPos, encVideoNeg, encAudioNeg,
                    guidance, audioGuidance, audioRescale, unguided, steps, steps + refineSteps, videoChannels,
                    onProgress, "s2 ", ancestral, seed);
            }
            finally
            {
                _transformer.StepGraphSuspended = false;
            }
        }

        Backend.Sync();
        // The captured graph bakes the DiT weight pointers; invalidate before any FreeWeights below so the next
        // generation re-warms and re-captures instead of replaying against freed memory.
        _transformer.InvalidateStepGraph(Backend);
        phase.Restart();
        stream.Dispose();
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
            _prefixPin.Resident = true;
            Logs.Info($"[ltx2-phase] resident prefix kept across generations: {residentBlocks} blocks " +
                $"(KEEP_MODELS; free {Backend.FreeMemoryBytes() >> 20} MB for decode)");
        }
        else
        {
            // Frees shared weights + the resident prefix (streamed blocks are already evicted — FreeWeights
            // skips non-cached tensors). The VAE decode needs this VRAM back.
            Backend.FreeWeights(_transformer.EnumerateWeights());
            _prefixPin.Resident = false;
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
        if (_diffusionVae is not null && _prefixPin.Resident && ReferenceEquals(VaeBackend, Backend))
        {
            Logs.Info($"[ltx2-phase] dropping the resident prefix for the diffusion VAE decode "
                + $"(free {Backend.FreeMemoryBytes() >> 20} MB before).");
            Backend.FreeWeights(_transformer.EnumerateWeights());
            long freeAfterDrop = Backend.FreeMemoryBytes();
            Backend.TrimMemoryPool();
            Logs.Info($"[ltx2-phase] diffusion-VAE prefix drop: free {freeAfterDrop >> 20} MB after FreeWeights, "
                + $"{Backend.FreeMemoryBytes() >> 20} MB after TrimMemoryPool.");
            _prefixPin.Resident = false;
        }
        else if (_prefixPin.Resident && ReferenceEquals(VaeBackend, Backend) && Backend.FreeMemoryBytes() < decodeNeed)
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
                _prefixPin.Resident = false;
            }
            else _prefixPin.TailEvicted = true;
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
        byte[][] frames = VideoRgbFrames.ExtractAllFrames(rgb);
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

    /// <summary>Two-stage latent grid for a requested pixel size: stage 1 is the request HALVED and snapped down to
    /// the latent grid, stage 2 is that doubled. The halving is what can lose a cell — a 1280x736 request denoises
    /// at 20x11 and renders 1280x704, one latent row short of the request, because 736/2 = 368 is not a whole number
    /// of 32-px cells. Returns (stage-1 h, stage-1 w, final h, final w) in LATENT cells.</summary>
    internal static (int Stage1Height, int Stage1Width, int Height, int Width) TwoStageGrid(
        int width, int height, int spatialCompression)
    {
        int h1 = height / 2 / spatialCompression, w1 = width / 2 / spatialCompression;
        if (h1 < 1 || w1 < 1)
            throw new ArgumentException($"LTX-2.5 two-stage needs at least {spatialCompression * 2} px on each axis; got {width}x{height}.");
        return (h1, w1, h1 * 2, w1 * 2);
    }

    /// <summary>One flow-match denoise stage, in place on both latents. <paramref name="sigmas"/> has one more entry
    /// than the stage has steps; <paramref name="stepBase"/>/<paramref name="stepTotal"/> place those steps inside
    /// the generation's overall progress, and <paramref name="stageTag"/> labels its log lines.</summary>
    private void DenoiseStage(Tensor videoLat, Tensor audioLat, float[] sigmas,
        (int Frames, int Height, int Width) grid, int audioFrames, double frameRate,
        Tensor encVideoPos, Tensor encAudioPos, Tensor encVideoNeg, Tensor encAudioNeg,
        float guidance, float audioGuidance, float audioRescale, bool unguided,
        int stepBase, int stepTotal, int videoChannels, Action<GenerationProgress>? onProgress, string stageTag,
        bool ancestral, int seed)
    {
        int steps = sigmas.Length - 1;
        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            (float dt, float zScale, float noiseScale) = ancestral ? AncestralCoefficients(sigmas[k], sigmas[k + 1])
                : (sigmas[k] - sigmas[k + 1], 1f, 0f);
            float tEmb = sigmas[k] * _config.TimestepScaleMultiplier;   // flow sigma (≈1..0) scaled to ≈0..1000

            Tensor vCondV, vCondA, vUncondV, vUncondA;
            bool paired = !unguided;
            if (unguided)
            {
                // At guidance 1 the pair reduces to the conditional branch, so running the unconditional one is pure
                // waste — half the DiT work per step, which is most of the step. The distilled schedule always
                // lands here.
                (vCondV, vCondA) = _transformer.Forward(Backend, videoLat, audioLat, encVideoPos, encAudioPos,
                    tEmb, grid, audioFrames, frameRate, null, null);
                vUncondV = vCondV;
                vUncondA = vCondA;
            }
            else
            {
                // CFG-paired forward: both branches share each block's (streamed) weights within the step — half the
                // weight traffic of two sequential forwards.
                ((vCondV, vCondA), (vUncondV, vUncondA)) = _transformer.ForwardCfgPair(
                    Backend, videoLat, audioLat, encVideoPos, encAudioPos, encVideoNeg, encAudioNeg,
                    tEmb, grid, audioFrames, frameRate);
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
            if (noiseScale > 0f)
            {
                // Fresh noise per step per stream, seeded off the generation seed so a repeat reproduces.
                AncestralRenoise(videoLat, zScale, noiseScale, seed ^ 0x71A3 ^ (stepBase + k));
                AncestralRenoise(audioLat, zScale, noiseScale, seed ^ 0x71C5 ^ (stepBase + k));
            }

            Backend.Sync();
            sw.Stop();
            Logs.Info($"[ltx2-phase] {stageTag}step {stepBase + k + 1}/{stepTotal}: {sw.ElapsedMilliseconds} ms "
                + (paired ? "(paired CFG)" : "(single branch)"));
            // The preview drain reads the latent on host, which EVICTS it from the GPU — incompatible with the step
            // graph, whose capture bakes the resident latent's device address (a re-upload would move it). Skip the
            // preview while the graph is active (correctness > the optional live thumbnail).
            if (onProgress is not null && !_transformer.StepGraphActive)
            {
                Tensor preview = ExtractMiddleFrame(videoLat, grid.Frames, grid.Height, grid.Width, videoChannels);
                onProgress.Invoke(new GenerationProgress(stepBase + k + 1, stepTotal, sw.Elapsed.TotalMilliseconds)
                {
                    Latent = preview,
                    LatentArch = LatentArchitecture.Ltx,
                });
                preview.Dispose();
            }
            // Window the op profiler onto the steady-state steps: step 1 carries the int8 row-scale upload storm,
            // and everything before the loop is text encode. No-op when profiling is off.
            if (k == 0) Backend.ResetOpProfile();
        }
        Backend.DumpOpProfile($"denoise{stageTag.Trim()}{Math.Max(1, steps - 1)}");
    }

    /// <summary>Ancestral-Euler (eta=1) step coefficients for a sigma pair, from ComfyUI's
    /// <c>sample_euler_ancestral_RF</c>: the deterministic Euler step goes to <c>sigma_down = s1²/s0</c> instead of
    /// <c>s1</c>, then <c>z = zScale·z + noiseScale·noise</c> carries it back up. By construction
    /// <c>(zScale·sigma_down)² + noiseScale² = s1²</c>, which is what makes the injection marginal-preserving.
    /// The terminal pair (<c>s1 = 0</c>) degenerates to plain Euler: zScale 1, noiseScale 0.</summary>
    internal static (float Delta, float ZScale, float NoiseScale) AncestralCoefficients(float s0, float s1)
    {
        float sigmaDown = s0 <= 0f ? 0f : s1 * s1 / s0;
        float zScale = (1f - s1) / (1f - sigmaDown);
        float noiseScale = MathF.Sqrt(MathF.Max(0f, s1 * s1 - sigmaDown * sigmaDown * zScale * zScale));
        return (s0 - sigmaDown, zScale, noiseScale);
    }

    /// <summary>Ancestral noise injection, in place on device: <c>z = zScale·z + noiseScale·noise</c>. Composed from
    /// <c>AffineMix</c> + <c>CopyInto</c> rather than a new kernel — both are existing device ops, and the pair
    /// keeps the latent's tensor identity (and its device buffer) across the step.</summary>
    private void AncestralRenoise(Tensor z, float zScale, float noiseScale, int seed)
    {
        using Tensor noise = SeedGenerator.CreateNoise(z.Shape, seed);
        using Tensor mixed = new Tensor(z.Shape, DType.F32);
        Backend.AffineMix(mixed, z, noise, zScale, noiseScale);
        Backend.CopyInto(z, mixed);
    }

    /// <summary>Stage-1 → stage-2 handoff. Un-normalizes the video latent, runs the learned x2 latent upsampler
    /// (which is defined on UN-normalized latents), re-normalizes, repacks at the doubled grid, and re-noises BOTH
    /// streams to <paramref name="sigma"/>. The audio latent is CARRIED at its own rate — it is never upsampled,
    /// only re-noised alongside. Caller owns the returned tensors and still owns the inputs.</summary>
    private (Tensor Video, Tensor Audio) UpsampleAndRenoise(Tensor videoLat, Tensor audioLat,
        int tLat, int hLat, int wLat, int videoChannels, float sigma, int seed,
        Action? releaseDitWeights, Action? restoreDitWeights)
    {
        Stopwatch phase = Stopwatch.StartNew();
        Tensor packed = UnpackVideoLatents(videoLat, tLat, hLat, wLat, videoChannels);
        Tensor denorm = LtxVideo2VaeDecoder.DenormalizeLatent(packed, _videoLatentsMean, _videoLatentsStd);
        packed.Dispose();
        ProbeTensor("two-stage upsampler input (denormalized)", denorm);

        long upsamplerBytes = 0;
        foreach (Tensor t in LatentUpsampler!.EnumerateWeights())
            upsamplerBytes += t.DType.ComputeByteCount(t.ElementCount);
        Backend.TrimMemoryPool();
        // The upsampler is F32-only (Conv3d has no lower-precision path), so its ~2 GB has to fit beside whatever
        // the DiT parked. Displace the DiT rather than OOM — restoreDitWeights re-uploads before the refine stage.
        long need = upsamplerBytes + (1L << 30);
        bool displacedDit = false;
        if (releaseDitWeights is not null && restoreDitWeights is not null && Backend.FreeMemoryBytes() < need)
        {
            Logs.Info($"[ltx2-two-stage] releasing the resident DiT weights for the upsampler (free "
                + $"{Backend.FreeMemoryBytes() >> 20} MB < {need >> 20} MB needed); re-uploaded before the refine stage.");
            releaseDitWeights();
            Backend.TrimMemoryPool();
            displacedDit = true;
        }
        Backend.PreloadWeights(LatentUpsampler.EnumerateWeights());
        Tensor upsampled = LatentUpsampler.Forward(Backend, denorm);
        denorm.Dispose();
        Backend.Sync();
        Backend.FreeWeights(LatentUpsampler.EnumerateWeights());
        Backend.TrimMemoryPool();
        if (displacedDit) restoreDitWeights!.Invoke();
        Logs.Info($"[ltx2-two-stage] latent upsample {hLat}x{wLat} -> {hLat * 2}x{wLat * 2}: {phase.ElapsedMilliseconds} ms"
            + $" ({upsamplerBytes >> 20} MB of F32 weights)");
        ProbeTensor("two-stage upsampler output (un-normalized)", upsampled);

        Tensor renormalized = LtxVideo2VaeDecoder.NormalizeLatent(upsampled, _videoLatentsMean, _videoLatentsStd);
        upsampled.Dispose();
        Tensor videoTokens = PackVideoLatents(renormalized, videoChannels);
        renormalized.Dispose();

        // Both latents re-enter the schedule at the refine stage's first sigma. Salts follow the existing
        // convention (video latent `seed`, audio `seed ^ 0x5D2B`, diffusion-decoder noise `seed ^ 0x2D5B`).
        Backend.Sync();
        using Tensor videoNoise = SeedGenerator.CreateNoise(videoTokens.Shape, seed ^ 0x6C41);
        using Tensor audioNoise = SeedGenerator.CreateNoise(audioLat.Shape, seed ^ 0x6C42);
        Tensor renoisedVideo = Renoise(Backend, videoTokens, videoNoise, sigma);
        videoTokens.Dispose();
        Tensor renoisedAudio = Renoise(Backend, audioLat, audioNoise, sigma);
        return (renoisedVideo, renoisedAudio);
    }

    /// <summary>Flow-match re-noise into a FRESH tensor: <c>sigma·noise + (1−sigma)·x</c> (ComfyUI
    /// <c>ModelSamplingDiscreteFlow.noise_scaling</c>) via <see cref="IBackend.AffineMix"/> — the same op
    /// <see cref="AncestralRenoise"/> uses. Out-of-place because writing a device-resident latent host-side
    /// leaves its GPU cache stale.</summary>
    internal static Tensor Renoise(IBackend backend, Tensor x, Tensor noise, float sigma)
    {
        Tensor outT = new Tensor(x.Shape, DType.F32);
        backend.AffineMix(outT, noise, x, sigma, 1f - sigma);
        return outT;
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

        // States and width come from the tower and DiT config so a variant mismatch fails loudly in the
        // connector shapes instead of silently mis-relayouting here.
        int states = _gemma.NumLayers + 1;              // 0=embeddings, 1..NumLayers=post-layer
        int captionChannels = _config.CaptionChannels;
        int[] layerIndices = new int[states];
        for (int i = 0; i < states; i++) layerIndices[i] = i;
        Stopwatch sub = Stopwatch.StartNew();
        Tensor multi = _gemma.EncodeMultiLayer(TextEncoderBackend, [padded], layerIndices);  // [1, seq, states·channels] layer-outer
        Logs.Info($"[ltx2-phase]   gemma encode ({seq} tok): {sub.ElapsedMilliseconds} ms");
        sub.Restart();

        // Relayout to channel-outer (feature = channel·states + layer), which the connector consumes.
        Tensor feats = new Tensor(new TensorShape(seq, states * captionChannels), DType.F32);
        float* sp = (float*)multi.DataPointer;
        float* dp = (float*)feats.DataPointer;
        long stride = (long)states * captionChannels;
        for (int t = 0; t < seq; t++)
        {
            float* srcRow = sp + (long)t * stride;     // [layer·channels + channel]
            float* dstRow = dp + (long)t * stride;     // [channel·states + layer]
            for (int l = 0; l < states; l++)
                for (int c = 0; c < captionChannels; c++)
                    dstRow[(long)c * states + l] = srcRow[(long)l * captionChannels + c];
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

    /// <summary>Token count the shift formula is evaluated at: the real count, unless <see
    /// cref="LtxVideo2Config.ShiftMaxTokens"/> (or <c>HARTSY_LTX2_SHIFT_MAX_TOKENS</c>, which wins) caps it.</summary>
    internal static int ShiftTokens(int videoTokens, LtxVideo2Config config)
    {
        int cap = EngineKnobs.Ltx2ShiftMaxTokens.Value ?? config.ShiftMaxTokens;
        return cap > 0 && videoTokens > cap ? cap : videoTokens;
    }

    /// <summary>Dynamic flow-match shift (LTX-2 scheduler: base_seq 1024 → base_shift 0.95, max_seq 4096 →
    /// max_shift 2.05). The fit is calibrated on 1024→4096 tokens and <c>exp()</c> extrapolates: 27,280 tokens
    /// (1280x704x241f) gives shift 31,306, a schedule where 33 of 40 steps move sigma by a rounding error. Both
    /// references keep the shift resolution-INDEPENDENT — diffusers passes a constant 4096 seq len, and the
    /// distilled sigmas are token-count free — so <see cref="LtxVideo2Config.ShiftMaxTokens"/> caps the token
    /// count fed here (4096 reproduces diffusers' constant 7.768).</summary>
    internal static float ComputeShift(int videoTokens, LtxVideo2Config config)
    {
        float direct = EngineKnobs.Ltx2Shift.Value ?? config.ShiftOverride;
        return direct > 0f ? direct : FormulaShift(ShiftTokens(videoTokens, config));
    }

    /// <summary>The fit itself, uncapped — kept separate so the log can show what the knobs changed.</summary>
    internal static float FormulaShift(int tokens)
    {
        double m = (2.05 - 0.95) / (4096 - 1024), b = 0.95 - m * 1024;
        return (float)Math.Exp(tokens * m + b);
    }

    /// <summary>Test seam for <see cref="StretchTerminal"/> — the schedule is pure math and worth pinning against
    /// ComfyUI's numbers without standing up a pipeline.</summary>
    internal static float[] StretchTerminalForTests(float[] sigmas, LtxVideo2Config config, float shift)
        => StretchTerminal(sigmas, config, shift);

    /// <summary>Rescales a shifted flow-match schedule so its last non-zero sigma lands on the config's terminal
    /// value, matching LTX's own scheduler (<c>stretch=True, terminal=0.1</c> in the shipped templates). Without it
    /// the denoise stops early: the shift grows with token count, so at 1280x736x97f the schedule ends at sigma
    /// 0.817 and drops to zero in one step. <see cref="LtxVideo2Config.SigmaStretch"/> = false restores the
    /// un-stretched schedule (the =0-is-worse ablation is settled; the env kill-switch is gone).</summary>
    private static float[] StretchTerminal(float[] sigmas, LtxVideo2Config config, float shift)
    {
        if (!config.SigmaStretch)
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
        if (EngineKnobs.Ltx2AudioDump.Value is not { Length: > 0 } dir)
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
        Logs.Debug($"[ltx2-dump] wrote {name} {tensor.Shape} to {dir}");
    }

    /// <summary>Logs min/max/mean/rms for a stage output under <c>HARTSY_LTX2_PROBE=1</c>; no-op otherwise.</summary>
    private static void ProbeTensor(string label, Tensor tensor)
    {
        if (!EngineKnobs.Ltx2Probe.Value)
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
    internal static Tensor UnpackVideoLatents(Tensor tokens, int t, int h, int w, int channels)
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
            if (_prefixPin.Resident)
            {
                Backend.FreeWeights(_transformer.EnumerateWeights());
                Backend.TrimMemoryPool();
                _prefixPin.Resident = false;
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

    /// <summary>Packs a video latent <c>[1, C, T, H, W] → [T·H·W, C]</c> in (f,h,w) order — the exact inverse of
    /// <see cref="UnpackVideoLatents"/>.</summary>
    internal static Tensor PackVideoLatents(Tensor latent, int channels)
    {
        int t = (int)latent.Shape[2], h = (int)latent.Shape[3], w = (int)latent.Shape[4];
        long spatial = (long)t * h * w;
        Tensor outT = new Tensor(new TensorShape((int)spatial, channels), DType.F32);
        float* sp = (float*)latent.DataPointer;
        float* dp = (float*)outT.DataPointer;
        for (int ti = 0; ti < t; ti++)
            for (int hi = 0; hi < h; hi++)
                for (int wi = 0; wi < w; wi++)
                {
                    long token = ((long)ti * h + hi) * w + wi;
                    for (int c = 0; c < channels; c++)
                        dp[token * channels + c] = sp[(long)c * spatial + token];
                }
        return outT;
    }

    /// <summary>Middle latent frame <c>[1, C, H, W]</c> for latent2rgb previews.</summary>
    internal static Tensor ExtractMiddleFrame(Tensor tokens, int t, int h, int w, int channels)
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
