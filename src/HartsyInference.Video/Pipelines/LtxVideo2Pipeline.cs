using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
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

/// <summary>LTX-2.3 (Lightricks, 22B) text-to-video+audio pipeline. Drives the dual-stream
/// <see cref="LtxVideo2Transformer"/> end-to-end: Gemma-3-12B all-49-layer features → per-modality
/// <see cref="LtxVideo2TextConnectors"/> → flow-match Euler denoise of the interleaved video+audio latent streams
/// (2-way text CFG) → <see cref="LtxVideo2VaeDecoder"/> for RGB and <see cref="LtxAudioVaeDecoder"/> +
/// <see cref="LtxAudioVocoder"/> for the waveform.
///
/// <para>Token packing follows LTX patch-1: the video latent <c>[1,128,T,H,W]</c> packs to <c>[T·H·W, 128]</c> in
/// (f,h,w) order and the audio latent <c>[1,8,L,16]</c> packs to <c>[L, 128]</c> (channel·16+mel). The dual-stream
/// DiT consumes both each step and returns both velocities; CFG is standard velocity-space (the reference's
/// velocity→x0→delta→velocity round-trip reduces to this when guidance-rescale / STG / modality-isolation are off,
/// which are the defaults). <b>Status: built end-to-end, first-run numeric validation pending</b> — the flow-match
/// shift, DiT timestep scaling, and audio latent sizing are validation-gated, consistent with the other LTX
/// pipelines.</para></summary>
public sealed unsafe class LtxVideo2Pipeline : DiffusionPipelineBase
{
    private const int GemmaCaptionChannels = 3840;
    private const int GemmaLayers = 49;             // 48 transformer layers + 1 embedding
    private const int ConnectorRegisters = 128;     // text seq is padded to a multiple of this

    private readonly LtxVideo2Transformer _transformer;
    private readonly LtxVideo2TextConnectors _connectors;
    private readonly LtxVideo2VaeDecoder _vae;
    private readonly LtxAudioVaeDecoder? _audioVae;
    private readonly LtxAudioVocoder? _vocoder;
    private readonly LlamaStyleEncoder _gemma;
    private readonly LtxVideo2Config _config;
    private readonly float[]? _audioLatentsMean, _audioLatentsStd;

    public LtxVideo2Pipeline(IBackend backend, LtxVideo2Transformer transformer, LtxVideo2TextConnectors connectors,
        LtxVideo2VaeDecoder vae, LlamaStyleEncoder gemma, LtxVideo2Config config,
        LtxAudioVaeDecoder? audioVae = null, LtxAudioVocoder? vocoder = null,
        float[]? audioLatentsMean = null, float[]? audioLatentsStd = null)
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
        float guidance = request.CfgScale ?? _config.GuidanceScale;

        // Dynamic flow-match shift (LTX-2 scheduler: base_seq 1024 → base_shift 0.95, max_seq 4096 → max_shift 2.05).
        double m = (2.05 - 0.95) / (4096 - 1024), bShift = 0.95 - m * 1024;
        float shift = (float)Math.Exp(sv * m + bShift);

        Logs.Info($"LTX-2 T2V+A: {numFrames}f {width}x{height}, {steps} steps, cfg={guidance}, " +
            $"seed={seed} (video {tLat}x{hLat}x{wLat}={sv} tokens, audio {audioFrames} tokens, shift={shift:F3})");
        Logs.Warning("LTX-2 pipeline is first-run-validation pending — numerics unverified vs the reference checkpoint.");

        // 1. Text conditioning (run once): Gemma 49-layer features → per-modality connector embeddings.
        (Tensor encVideoPos, Tensor encAudioPos) = EncodeText(promptTokens);
        (Tensor encVideoNeg, Tensor encAudioNeg) = EncodeText(negativeTokens);

        // Reclaim the ~12 GB Gemma encoder before the DiT — both can't be resident on 24 GB.
        Backend.Sync();
        Backend.FreeWeights(_gemma.EnumerateWeights());

        // 2. Denoise both streams. The 22B fp8 DiT (~22 GB) doesn't fit resident alongside activations on 24 GB, so
        // stream its 48 blocks on/off device (only the shared modulation tables stay resident). CPU/Vulkan (no
        // streaming cache) preload everything eagerly.
        HartsyInference.Core.MemoryManagement.BlockStreamingController? streamer = null;
        if (Backend.StreamingCache is not null)
        {
            Backend.PreloadWeights(_transformer.EnumerateSharedWeights());
            HartsyInference.Core.MemoryManagement.IStreamingBlock[] blocks =
                new HartsyInference.Core.MemoryManagement.IStreamingBlock[_transformer.BlockCount];
            for (int b = 0; b < blocks.Length; b++) blocks[b] = _transformer.GetBlock(b);
            streamer = new HartsyInference.Core.MemoryManagement.BlockStreamingController(Backend.StreamingCache, blocks, prefetchAhead: 2, retainBehind: 0);
            _transformer.BeforeBlockForward = streamer.BeforeBlockForward;
            streamer.Prime();
            Logs.Info($"LTX-2 streaming: {blocks.Length} blocks, ~{streamer.EstimatedTotalWeightBytes / (1024 * 1024)} MB total (resident window ~{streamer.EstimatedTotalWeightBytes / blocks.Length * 3 / (1024 * 1024)} MB)");
        }
        else
        {
            Backend.PreloadWeights(_transformer.EnumerateWeights());
        }
        Tensor videoLat = SeedGenerator.CreateNoise(new TensorShape(sv, videoChannels), seed);
        Tensor audioLat = SeedGenerator.CreateNoise(new TensorShape(audioFrames, audioChannels), seed ^ 0x5D2B);
        float[] tsteps = LancePipelineCommon.BuildShiftedTimesteps(steps, shift);

        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float dt = tsteps[k] - tsteps[k + 1];
            float sigma = tsteps[k];                                    // raw flow sigma (≈1..0), for prompt_adaln
            float tEmb = sigma * _config.TimestepScaleMultiplier;       // ≈0..1000, for the other modulators

            (Tensor vCondV, Tensor vCondA) = _transformer.Forward(Backend, videoLat, audioLat, encVideoPos, encAudioPos,
                tEmb, (tLat, hLat, wLat), audioFrames, frameRate, null, null, sigma);
            (Tensor vUncondV, Tensor vUncondA) = _transformer.Forward(Backend, videoLat, audioLat, encVideoNeg, encAudioNeg,
                tEmb, (tLat, hLat, wLat), audioFrames, frameRate, null, null, sigma);

            LancePipelineCommon.EulerCfgStep(videoLat, vCondV, vUncondV, guidance, dt);
            LancePipelineCommon.EulerCfgStep(audioLat, vCondA, vUncondA, guidance, dt);
            vCondV.Dispose(); vCondA.Dispose(); vUncondV.Dispose(); vUncondA.Dispose();

            sw.Stop();
            if (onProgress is not null)
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
        if (streamer is not null) { _transformer.BeforeBlockForward = null; streamer.EvictAll(); streamer.Dispose(); Backend.FreeWeights(_transformer.EnumerateSharedWeights()); }
        else Backend.FreeWeights(_transformer.EnumerateWeights());
        encVideoPos.Dispose(); encAudioPos.Dispose(); encVideoNeg.Dispose(); encAudioNeg.Dispose();

        // 3. Decode video.
        Tensor videoVaeLatent = UnpackVideoLatents(videoLat, tLat, hLat, wLat, videoChannels);
        videoLat.Dispose();
        Tensor rgb = _vae.Decode(Backend, videoVaeLatent);
        videoVaeLatent.Dispose();
        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = VideoRgbFrames.ExtractFrame(rgb, i);
        rgb.Dispose();

        // 4. Decode audio (optional — requires the audio VAE + vocoder).
        float[][]? audio = null;
        int audioSampleRate = 0;
        if (_audioVae is not null && _vocoder is not null)
            audio = DecodeAudio(audioLat, audioFrames, out audioSampleRate);
        audioLat.Dispose();

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

        // Right-pad to a register multiple. The Gemma encoder applies only a causal mask (no padding mask), so
        // padding on the right keeps real tokens (at the front) from attending to pad tokens; validMask marks them.
        int[] padded = new int[seq];
        for (int i = 0; i < real; i++) padded[i] = tokens[i];
        float[] validMask = new float[seq];
        for (int i = 0; i < real; i++) validMask[i] = 1f;

        int[] layerIndices = new int[GemmaLayers];
        for (int i = 0; i < GemmaLayers; i++) layerIndices[i] = i;       // 0=embeddings, 1..48=post-layer
        Tensor multi = _gemma.EncodeMultiLayer(Backend, [padded], layerIndices);  // [1, seq, 49·3840] layer-outer

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

        (Tensor video, Tensor audio) = _connectors.Forward(Backend, feats, validMask);
        feats.Dispose();
        return (video, audio);
    }

    /// <summary>Audio VAE + vocoder: denormalize → unpack <c>[L,128]→[1,8,L,16]</c> → mel → 48 kHz waveform.</summary>
    private float[][] DecodeAudio(Tensor audioLat, int audioFrames, out int sampleRate)
    {
        Backend.PreloadWeights(_audioVae!.EnumerateWeights());

        int latentChannels = 8;
        int melLat = _config.AudioInChannels / latentChannels;   // 128 / 8 = 16
        Tensor unpacked = UnpackAudioLatents(audioLat, audioFrames, latentChannels, melLat);
        Tensor mel = _audioVae.Decode(Backend, unpacked);        // [1, 2, T, 64]
        unpacked.Dispose();
        // The vocoder manages its own weights (no bulk EnumerateWeights); ops fault them in on demand.
        Tensor wave = _vocoder!.Forward(Backend, mel);           // [1, channels, samples]
        mel.Dispose();
        Backend.Sync();
        Backend.FreeWeights(_audioVae.EnumerateWeights());

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
