using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>Wan2.2-S2V (speech-to-video) pipeline — single-chunk. Drives the <see cref="WanS2VTransformer"/>:
/// per-frame Wav2Vec2 audio features are encoded (<see cref="WanS2VAudioEncoder"/>) into audio tokens that the DiT's
/// audio injector cross-attends to, alongside umT5 text. Flow-match Euler + 2-way text CFG; reuses the Wan2.1 z=16 VAE.
///
/// <para><b>Status:</b> structural, reconstructed (no diffusers reference). The <b>audio features are pre-computed</b>
/// (the Wav2Vec2 front-end is the deferred Phase S0), the <b>reference-image identity conditioning and the
/// autoregressive multi-chunk long-video loop are NOT modeled here</b> (single clip, audio+text only). Numerics +
/// structure validation-pending.</para></summary>
public sealed unsafe class WanS2VPipeline : DiffusionPipelineBase
{
    private readonly WanS2VTransformer _transformer;
    private readonly WanS2VAudioEncoder _audioEncoder;
    private readonly IWanVaeDecoder _vae;
    private readonly WanVideoConfig _config;

    public WanS2VPipeline(IBackend backend, WanS2VTransformer transformer, WanS2VAudioEncoder audioEncoder,
        IWanVaeDecoder vae, WanVideoConfig config)
        : base(backend)
    {
        _transformer = transformer;
        _audioEncoder = audioEncoder;
        _vae = vae;
        _config = config;
    }

    /// <summary>Generates a clip from per-frame audio features. <paramref name="audioFeatures"/> is the Wav2Vec2
    /// stacked-layer features resampled to the latent frame rate: <c>[gt, numLayers, audioDim]</c> where
    /// <c>gt = (numFrames-1)/tp + 1</c> (one audio group per latent frame).</summary>
    public (byte[][] frames, int width, int height, int seed) GenerateFromAudioFeatures(
        Tensor promptEmbeds, Tensor negativeEmbeds, Tensor audioFeatures, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        if (request.Width % sp != 0 || request.Height % sp != 0)
            throw new ArgumentException($"Width/height must be divisible by {sp} for Wan-S2V.");
        if (numFrames < 1 || (numFrames - 1) % tp != 0)
            throw new ArgumentException($"num_frames must satisfy (num_frames-1) % {tp} == 0; got {numFrames}.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int tLat = (numFrames - 1) / tp + 1, hLat = request.Height / sp, wLat = request.Width / sp, latentCh = _config.VaeLatentChannels;
        if ((int)audioFeatures.Shape[0] != tLat)
            throw new ArgumentException($"audioFeatures must have {tLat} frame groups (latent frames); got {audioFeatures.Shape[0]}.", nameof(audioFeatures));
        int steps = request.Steps > 0 ? request.Steps : _config.NumInferenceSteps;
        float guidance = request.CfgScale > 0 ? request.CfgScale : _config.GuidanceScale;
        float shift = _config.FlowShift;

        Logs.Info($"Wan-S2V: {numFrames}f {request.Width}x{request.Height}, {steps} steps, cfg={guidance}, seed={seed} " +
            $"(latent {latentCh}x{tLat}x{hLat}x{wLat})");
        Logs.Warning("Wan-S2V pipeline is reconstructed + first-run-validation pending — reference/multi-chunk not modeled.");

        // Encode the audio once (it's fixed across denoise steps).
        Tensor audioTokens = _audioEncoder.Forward(Backend, audioFeatures);   // [gt, tokens, dim]

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        Tensor latents = SeedGenerator.CreateNoise(new TensorShape([1L, latentCh, tLat, hLat, wLat]), seed);
        float[] tsteps = LancePipelineCommon.BuildShiftedTimesteps(steps, shift);

        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float t = tsteps[k], dt = t - tsteps[k + 1], tEmb = t * 1000f;
            Tensor vCond = _transformer.Forward(Backend, latents, audioTokens, promptEmbeds, tEmb);
            Tensor vUncond = _transformer.Forward(Backend, latents, audioTokens, negativeEmbeds, tEmb);
            LancePipelineCommon.EulerCfgStep(latents, vCond, vUncond, guidance, dt);
            vCond.Dispose(); vUncond.Dispose();
            sw.Stop();
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
            {
                Latent = latents,
                LatentArch = LatentArchitecture.Wan,
            });
        }

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());
        audioTokens.Dispose();

        Tensor rgb;
        try { rgb = _vae.Decode(Backend, latents); }
        finally { latents.Dispose(); }
        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = VideoRgbFrames.ExtractFrame(rgb, i);
        rgb.Dispose();
        Logs.Info($"Wan-S2V complete ({frames.Length} frames, seed={seed})");
        return (frames, request.Width, request.Height, seed);
    }
}
