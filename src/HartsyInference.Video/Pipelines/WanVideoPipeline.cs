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

/// <summary>Wan-Video (Wan-AI, Apache-2.0) text-to-video pipeline — Wan2.2 TI2V-5B. Maximum reuse: the DiT denoises directly in VAE-latent space <c>[1,48,T,H,W]</c> (the transformer patchifies/unpatchifies internally), and the VAE is the **already-built <see cref="Wan22VaeDecoder"/>** (z=48, 16×/4×, streaming). Flow-match Euler + 2-way text CFG; reuses <see cref="LancePipelineCommon"/> + frame streaming.
///
/// <para>Takes pre-computed umT5 features (encode upstream with the shared T5 encoder). <c>T_lat = (num_frames−1)/4 + 1</c>; latent <c>H/16 × W/16</c>. <b>Status: built, first-run validation pending</b> — the flow-match shift (5.0/3.0), scheduler (UniPC vs Euler), and DiT timestep scaling are validation-gated.</para></summary>
public sealed unsafe class WanVideoPipeline : DiffusionPipelineBase
{
    private readonly WanVideoTransformer _transformer;
    private readonly Wan22VaeDecoder _vae;
    private readonly Wan22VaeEncoder? _encoder;
    private readonly WanVideoConfig _config;

    /// <summary><paramref name="encoder"/> is optional and only needed for RGB-input I2V (<see cref="EncodeFirstFrame"/>);
    /// it loads from the same <c>wan22_vae.safetensors</c> dict as the decoder.</summary>
    public WanVideoPipeline(IBackend backend, WanVideoTransformer transformer, Wan22VaeDecoder vae, WanVideoConfig config,
        Wan22VaeEncoder? encoder = null)
        : base(backend)
    {
        _transformer = transformer;
        _vae = vae;
        _config = config;
        _encoder = encoder;
    }

    /// <summary>Encodes an interleaved-RGB24 conditioning frame to the normalized first-frame latent for the TI2V
    /// I2V path — pass the result as <c>firstFrameLatent</c> to <see cref="GenerateFromEmbeddings"/> /
    /// <see cref="GenerateFramesAsync"/>. The caller owns (disposes) the returned tensor. Requires the pipeline to be
    /// constructed with a <see cref="Wan22VaeEncoder"/>.</summary>
    public Tensor EncodeFirstFrame(ReadOnlySpan<byte> rgb24, int width, int height)
    {
        ThrowIfDisposed();
        if (_encoder is null)
            throw new InvalidOperationException("RGB-input I2V needs a Wan22VaeEncoder — construct the pipeline with one (it loads from the same VAE weights).");
        return _encoder.EncodeRgbFrame(Backend, rgb24, width, height);
    }

    /// <summary>Generates frames from pre-computed umT5 features <c>[L, textDim]</c>. Returns one interleaved-RGB <c>byte[]</c> per frame.
    /// <para><paramref name="firstFrameLatent"/> switches to the TI2V image-to-video path (diffusers
    /// <c>expand_timesteps</c>): a <c>[1, 48, 1, H/16, W/16]</c> VAE-encoded <b>and latent-normalized</b>
    /// (<see cref="Wan22VaeLatentNorm.Normalize"/>) first frame that is re-imposed into the model input each step at
    /// per-frame timestep 0 while the remaining frames denoise. The Wan2.2 VAE <i>encoder</i> is not built yet —
    /// produce the conditioning latent offline (validation-gated).</para></summary>
    public (byte[][] frames, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor promptEmbeds, Tensor negativeEmbeds, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress = null, Tensor? firstFrameLatent = null)
    {
        Tensor latent = RunDenoise(promptEmbeds, negativeEmbeds, request, numFrames, onProgress, firstFrameLatent, out int seed);
        Tensor rgb;
        try { rgb = _vae.Decode(Backend, latent); }
        finally { latent.Dispose(); }

        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = FrameToBytes(rgb, i);
        rgb.Dispose();
        Logs.Info($"Wan-Video T2V complete ({frames.Length} frames, seed={seed})");
        return (frames, request.Width, request.Height, seed);
    }

    /// <summary>Streams decoded frames (pull-based → memory bounded; pair with an <c>IVideoEncoder</c>).
    /// <paramref name="firstFrameLatent"/> enables the TI2V image-to-video path — see <see cref="GenerateFromEmbeddings"/>.</summary>
    public async IAsyncEnumerable<VideoFrame> GenerateFramesAsync(
        Tensor promptEmbeds, Tensor negativeEmbeds, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress = null, Tensor? firstFrameLatent = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Tensor latent = RunDenoise(promptEmbeds, negativeEmbeds, request, numFrames, onProgress, firstFrameLatent, out _);
        try
        {
            int idx = 0;
            foreach (Tensor group in _vae.DecodeStreaming(Backend, latent))   // [1,3,groupT,H,W] per latent frame
            {
                int gT = (int)group.Shape[2], h = (int)group.Shape[3], w = (int)group.Shape[4];
                for (int gi = 0; gi < gT; gi++)
                    yield return new VideoFrame(idx++, w, h, FrameToBytesGroup(group, gi));
                group.Dispose();
                cancellationToken.ThrowIfCancellationRequested();
                await Task.Yield();
            }
        }
        finally { latent.Dispose(); }
    }

    /// <summary>Runs the flow-match denoise loop in (normalized) latent space and returns <c>[1,48,T_lat,H_lat,W_lat]</c>.
    /// With <paramref name="firstFrameLatent"/> set, follows the diffusers <c>expand_timesteps</c> I2V path: the model
    /// input gets the condition imposed on frame 0 with per-frame timestep 0 each step, the evolving latents are
    /// stepped freely, and the condition is re-imposed once after the loop.</summary>
    private Tensor RunDenoise(Tensor promptEmbeds, Tensor negativeEmbeds, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress, Tensor? firstFrameLatent, out int seed)
    {
        ThrowIfDisposed();
        seed = request.Seed ?? SeedGenerator.RandomSeed();
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        if (request.Width % sp != 0 || request.Height % sp != 0)
            throw new ArgumentException($"Width/height must be divisible by {sp} for Wan-Video.");
        if (numFrames < 1 || (numFrames - 1) % tp != 0)
            throw new ArgumentException($"num_frames must satisfy (num_frames-1) % {tp} == 0; got {numFrames}.");

        int tLat = (numFrames - 1) / tp + 1;
        int hLat = request.Height / sp, wLat = request.Width / sp;
        if (firstFrameLatent is not null &&
            (firstFrameLatent.Shape.Rank != 5 || firstFrameLatent.Shape[0] != 1 || firstFrameLatent.Shape[1] != _config.InChannels
             || firstFrameLatent.Shape[2] != 1 || firstFrameLatent.Shape[3] != hLat || firstFrameLatent.Shape[4] != wLat))
            throw new ArgumentException($"firstFrameLatent must be [1,{_config.InChannels},1,{hLat},{wLat}]; got {firstFrameLatent.Shape}.", nameof(firstFrameLatent));

        int steps = request.Steps > 0 ? request.Steps : _config.NumInferenceSteps;
        float guidance = request.CfgScale > 0 ? request.CfgScale : _config.GuidanceScale;
        float shift = _config.FlowShift;

        string mode = firstFrameLatent is null ? "T2V" : "I2V";
        Logs.Info($"Wan-Video {mode}: {numFrames}f {request.Width}x{request.Height}, {steps} steps, cfg={guidance}, seed={seed} (latent {_config.InChannels}x{tLat}x{hLat}x{wLat}, shift={shift})");
        Logs.Warning("Wan-Video pipeline is first-run-validation pending — numerics unverified vs the reference checkpoint.");

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        Tensor latents = SeedGenerator.CreateNoise(new TensorShape([1L, _config.InChannels, tLat, hLat, wLat]), seed);
        float[] tsteps = LancePipelineCommon.BuildShiftedTimesteps(steps, shift);
        float[]? frameTs = firstFrameLatent is null ? null : new float[tLat];

        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float t = tsteps[k], dt = t - tsteps[k + 1];
            float tEmb = t * 1000f;   // DiT timestep scaling (validation-gated)
            Tensor vCond, vUncond;
            if (firstFrameLatent is null)
            {
                vCond = _transformer.Forward(Backend, latents, promptEmbeds, tEmb);
                vUncond = _transformer.Forward(Backend, latents, negativeEmbeds, tEmb);
            }
            else
            {
                // Model input: condition on frame 0 (timestep 0), evolving noise elsewhere (timestep t).
                Tensor modelInput = CloneLatents(latents);
                WriteFirstFrame(modelInput, firstFrameLatent);
                frameTs![0] = 0f;
                for (int f = 1; f < tLat; f++) frameTs[f] = tEmb;
                vCond = _transformer.Forward(Backend, modelInput, promptEmbeds, frameTs);
                vUncond = _transformer.Forward(Backend, modelInput, negativeEmbeds, frameTs);
                modelInput.Dispose();
            }
            LancePipelineCommon.EulerCfgStep(latents, vCond, vUncond, guidance, dt);
            vCond.Dispose();
            vUncond.Dispose();
            sw.Stop();
            // Latent is a borrowed reference for preview encoders (latent2rgb decodes the middle frame).
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
            {
                Latent = latents,
                LatentArch = LatentArchitecture.Wan,
            });
        }

        if (firstFrameLatent is not null) WriteFirstFrame(latents, firstFrameLatent);

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());
        return latents;
    }

    private static Tensor CloneLatents(Tensor latents)
    {
        Tensor o = new Tensor(latents.Shape, DType.F32);
        long bytes = latents.Shape.ElementCount * 4;
        Buffer.MemoryCopy((float*)latents.DataPointer, (float*)o.DataPointer, bytes, bytes);
        return o;
    }

    /// <summary>Overwrites latent frame 0 of <paramref name="latents"/> <c>[1,C,T,H,W]</c> with <paramref name="condition"/> <c>[1,C,1,H,W]</c>.</summary>
    private static void WriteFirstFrame(Tensor latents, Tensor condition)
    {
        int c = (int)latents.Shape[1], t = (int)latents.Shape[2];
        long frame = latents.Shape[3] * latents.Shape[4];
        float* lp = (float*)latents.DataPointer;
        float* cp = (float*)condition.DataPointer;
        for (int ci = 0; ci < c; ci++)
            Buffer.MemoryCopy(cp + ci * frame, lp + (long)ci * t * frame, frame * 4, frame * 4);
    }

    private static byte[] FrameToBytes(Tensor rgb, int frameIndex) => VideoRgbFrames.ExtractFrame(rgb, frameIndex);

    private static byte[] FrameToBytesGroup(Tensor group, int frameIndex) => VideoRgbFrames.ExtractFrame(group, frameIndex);
}
