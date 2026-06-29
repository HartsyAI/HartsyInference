using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>LTX-Video (Lightricks, Apache-2.0) text-to-video pipeline. Reuses the foundation: the <see cref="LtxVideoTransformer"/> DiT, the <see cref="LtxVideoVaeDecoder"/> (base 0.9 config), flow-match Euler + 2-way CFG (<see cref="LancePipelineCommon"/>), and frame streaming (<see cref="VideoFrame"/> / <c>IVideoEncoder</c>).
///
/// <para>Takes pre-computed T5-XXL features (the transformer cross-attends to T5 hidden states; encode upstream with the shared T5 encoder). Latent grid is <c>(T_lat, H/32, W/32)</c> with <c>T_lat = (num_frames−1)/8 + 1</c>; tokens are the 128-channel latent flattened in <c>(f,h,w)</c> order (transformer patch 1). 2-way text CFG (guidance 3), 50 steps, rectified-flow Euler. <b>Status: built, first-run validation pending</b> — the flow-match timestep shift (dynamic μ), the timestep scaling into the DiT, and RoPE interpolation are validation-gated.</para></summary>
public sealed unsafe class LtxVideoPipeline : DiffusionPipelineBase
{
    private readonly LtxVideoTransformer _transformer;
    private readonly LtxVideoVaeDecoder _vae;
    private readonly LtxVideoConfig _config;
    private readonly int _latentChannels;

    public LtxVideoPipeline(IBackend backend, LtxVideoTransformer transformer, LtxVideoVaeDecoder vae, LtxVideoConfig config)
        : base(backend)
    {
        _transformer = transformer;
        _vae = vae;
        _config = config;
        _latentChannels = config.InChannels;
    }

    /// <summary>Generates frames from pre-computed T5 features. <paramref name="promptEmbeds"/>/<paramref name="negativeEmbeds"/> are <c>[L, captionChannels]</c>; returns one interleaved-RGB <c>byte[]</c> per frame.</summary>
    public (byte[][] frames, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor promptEmbeds, Tensor negativeEmbeds, TextToImageRequest request, int numFrames, int frameRate = 25,
        Action<GenerationProgress>? onProgress = null)
    {
        Tensor latent = RunDenoise(promptEmbeds, negativeEmbeds, request, numFrames, frameRate, onProgress, out int seed);
        Tensor rgb;
        try { rgb = DecodeLatent(latent, seed); }
        finally { latent.Dispose(); }

        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = FrameToBytes(rgb, i);
        rgb.Dispose();
        Logs.Info($"LTX-Video T2V complete ({frames.Length} frames, seed={seed})");
        return (frames, request.Width ?? 768, request.Height ?? 512, seed);
    }

    /// <summary>Streams decoded frames (pull-based → memory bounded; pair with an <c>IVideoEncoder</c>).</summary>
    public async IAsyncEnumerable<VideoFrame> GenerateFramesAsync(
        Tensor promptEmbeds, Tensor negativeEmbeds, TextToImageRequest request, int numFrames, int frameRate = 25,
        Action<GenerationProgress>? onProgress = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Tensor latent = RunDenoise(promptEmbeds, negativeEmbeds, request, numFrames, frameRate, onProgress, out int seed);
        Tensor rgb;
        try { rgb = DecodeLatent(latent, seed); }
        finally { latent.Dispose(); }

        try
        {
            int f = (int)rgb.Shape[2], w = (int)rgb.Shape[4], h = (int)rgb.Shape[3];
            for (int i = 0; i < f; i++)
            {
                yield return new VideoFrame(i, w, h, FrameToBytes(rgb, i));
                cancellationToken.ThrowIfCancellationRequested();
                await Task.Yield();
            }
        }
        finally { rgb.Dispose(); }
    }

    /// <summary>Decodes the VAE-ready latent to RGB. For the base 0.9.0 VAE (<c>VaeTimestepConditioned=false</c>)
    /// this is a plain decode. For the timestep-conditioned VAE (0.9.1+ / 13B) it first blends Gaussian noise into the
    /// latent at <c>DecodeNoiseScale</c> and decodes at <c>DecodeTimestep</c>.</summary>
    private Tensor DecodeLatent(Tensor latent, int seed)
    {
        if (!_config.VaeTimestepConditioned)
            return _vae.Decode(Backend, latent);

        // VALIDATION-PENDING: verify vs diffusers LTXPipeline 0.9.7 — diffusers blends pre-decode:
        //   noise = randn_like(latents)
        //   latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise
        // then vae.decode(latents, timestep=decode_timestep). The 1000× timestep_scale_multiplier is applied
        // *inside* the decoder, so we pass the raw decode_timestep (0.05) here.
        float s = _config.DecodeNoiseScale;
        // Distinct, reproducible noise stream from the sampling latent noise (seed offset).
        Tensor noise = SeedGenerator.CreateNoise(latent.Shape, unchecked(seed ^ 0x5EED_DEC0));
        float* lp = (float*)latent.DataPointer;
        float* np = (float*)noise.DataPointer;
        long n = latent.Shape.ElementCount;
        for (long i = 0; i < n; i++) lp[i] = (1f - s) * lp[i] + s * np[i];
        noise.Dispose();

        Logs.Info($"LTX-Video timestep-conditioned VAE decode (t={_config.DecodeTimestep:F3}, noise_scale={s:F3})");
        return _vae.Decode(Backend, latent, _config.DecodeTimestep);
    }

    /// <summary>Runs the flow-match denoise loop and returns the VAE-ready latent <c>[1,128,T_lat,H_lat,W_lat]</c>.</summary>
    private Tensor RunDenoise(Tensor promptEmbeds, Tensor negativeEmbeds, TextToImageRequest request, int numFrames, int frameRate,
        Action<GenerationProgress>? onProgress, out int seed)
    {
        ThrowIfDisposed();
        seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? 768, height = request.Height ?? 512;
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        if (width % sp != 0 || height % sp != 0)
            throw new ArgumentException($"Width/height must be divisible by {sp} for LTX-Video.");
        if (numFrames < 1 || (numFrames - 1) % tp != 0)
            throw new ArgumentException($"num_frames must satisfy (num_frames-1) % {tp} == 0; got {numFrames}.");

        int tLat = (numFrames - 1) / tp + 1;
        int hLat = height / sp, wLat = width / sp;
        int s = tLat * hLat * wLat;
        int steps = request.Steps ?? _config.NumInferenceSteps;
        float guidance = request.CfgScale ?? _config.GuidanceScale;

        // RoPE interpolation scale = (vae_temporal/frame_rate, vae_spatial, vae_spatial).
        (double T, double H, double W) interp = ((double)tp / frameRate, sp, sp);
        // Dynamic flow-match shift: mu = seq·m + b (base_seq 256 → 0.5, max_seq 4096 → 1.16), shift = exp(mu).
        double m = (1.16 - 0.5) / (4096 - 256), bShift = 0.5 - m * 256;
        float shift = (float)Math.Exp(s * m + bShift);

        Logs.Info($"LTX-Video T2V: {numFrames}f {width}x{height}, {steps} steps, cfg={guidance}, seed={seed} (grid {tLat}x{hLat}x{wLat}, {s} tokens, shift={shift:F3})");
        Logs.Warning("LTX-Video pipeline is first-run-validation pending — numerics unverified vs the reference checkpoint.");

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        Tensor latents = SeedGenerator.CreateNoise(new TensorShape(s, _latentChannels), seed);
        float[] tsteps = LancePipelineCommon.BuildShiftedTimesteps(steps, shift);

        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float t = tsteps[k], dt = t - tsteps[k + 1];
            float tEmb = t * 1000f;   // DiT timestep scaling (validation-gated)
            Tensor vCond = _transformer.Forward(Backend, latents, promptEmbeds, tEmb, (tLat, hLat, wLat), interp, null);
            Tensor vUncond = _transformer.Forward(Backend, latents, negativeEmbeds, tEmb, (tLat, hLat, wLat), interp, null);
            LancePipelineCommon.EulerCfgStep(latents, vCond, vUncond, guidance, dt);
            vCond.Dispose();
            vUncond.Dispose();
            sw.Stop();
            if (onProgress is not null)
            {
                // The working latent is token-packed [S, C]; hand preview encoders an NCHW
                // middle-frame slice instead (small copy: h·w·128 floats). Owned here, disposed
                // after the callback returns — encoders must not retain it.
                Tensor previewFrame = ExtractMiddleFrame(latents, tLat, hLat, wLat, _latentChannels);
                onProgress.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
                {
                    Latent = previewFrame,
                    LatentArch = LatentArchitecture.Ltx,
                });
                previewFrame.Dispose();
            }
        }

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        Tensor vaeLatent = UnpackLatents(latents, tLat, hLat, wLat, _latentChannels);   // [1,C,T_lat,H_lat,W_lat]
        latents.Dispose();
        return vaeLatent;
    }

    /// <summary>Extracts the middle latent frame from packed tokens <c>[S, C]</c> (f,h,w order,
    /// channel-last) as an NCHW <c>[1, C, H, W]</c> tensor for latent2rgb previews. Caller owns the result.</summary>
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

    /// <summary>Unpacks tokens <c>[S, C]</c> (f,h,w order, channel-last) → <c>[1, C, T, H, W]</c> (inverse of the patch-1 pack).</summary>
    private static Tensor UnpackLatents(Tensor tokens, int t, int h, int w, int channels)
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

    private static byte[] FrameToBytes(Tensor rgb, int frameIndex) => VideoRgbFrames.ExtractFrame(rgb, frameIndex);
}
