using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Video.Pipelines;

/// <summary>Lance (ByteDance, Apache-2.0) text-to-video pipeline. Reuses the entire image stack — <see cref="LanceTransformer"/> (the forward pass is T-agnostic), <see cref="LanceLatentPatch"/>, <see cref="Wan22VaeDecoder"/> (streaming T&gt;1 decode), and <see cref="LancePipelineCommon"/> helpers — adding only the time axis.
///
/// <para>Deltas vs <c>LanceImagePipeline</c>: <c>timestep_shift = 4.0</c>; token grid is <c>(T_lat, H/32, W/32)</c> with positions varying over t; and the VAE decode streams across latent frames. <c>num_frames</c> must satisfy <c>(num_frames − 1) % 4 == 0</c> (the model's native 4× temporal compression, max 121), giving <c>T_lat = (num_frames − 1)/4 + 1</c>.</para>
///
/// <para><b>Status: built, first-run validation pending</b> — like the image pipeline, the numerics are checkpoint-gated. T2V uses 2-way text CFG (3-way vision CFG / video editing needs the ViT — deferred). Frame streaming + encoding are layered on in <c>SharpInference.Video/Streaming</c> + <c>/Encoding</c>.</para></summary>
public sealed unsafe class LanceVideoPipeline : DiffusionPipelineBase
{
    private const int VaeChannels = 48;

    private readonly LanceTransformer _transformer;
    private readonly Wan22VaeDecoder _vae;
    private readonly LanceConfig _config;

    public LanceVideoPipeline(IBackend backend, LanceTransformer transformer, Wan22VaeDecoder vae, LanceConfig config)
        : base(backend)
    {
        _transformer = transformer;
        _vae = vae;
        _config = config;
    }

    /// <summary>Generates <paramref name="numFrames"/> RGB frames from chat-templated prompt + negative-prompt token ids, returning all frames at once. For memory-bounded streaming use <see cref="GenerateFramesAsync"/>.</summary>
    public (byte[][] frames, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds, int[] negativeTokenIds, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress = null)
    {
        Tensor vaeLatent = RunDenoise(promptTokenIds, negativeTokenIds, request, numFrames, onProgress, out int seed);
        Tensor rgb;
        try { rgb = _vae.Decode(Backend, vaeLatent); }
        finally { vaeLatent.Dispose(); }

        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = FrameToBytes(rgb, i);
        rgb.Dispose();
        Logs.Info($"Lance T2V complete ({frames.Length} frames, seed={seed})");
        return (frames, request.Width, request.Height, seed);
    }

    /// <summary>Streams decoded frames one at a time (pull-based → memory bounded to a single VAE frame-group; natural backpressure). Pair with an <c>IVideoEncoder</c> to write an MP4/frame-sequence without holding the whole clip.</summary>
    public async IAsyncEnumerable<VideoFrame> GenerateFramesAsync(
        int[] promptTokenIds, int[] negativeTokenIds, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Tensor vaeLatent = RunDenoise(promptTokenIds, negativeTokenIds, request, numFrames, onProgress, out _);
        try
        {
            int idx = 0;
            foreach (Tensor group in _vae.DecodeStreaming(Backend, vaeLatent))   // [1,3,groupT,H,W] per latent frame
            {
                int gT = (int)group.Shape[2], h = (int)group.Shape[3], w = (int)group.Shape[4];
                for (int gi = 0; gi < gT; gi++)
                    yield return new VideoFrame(idx++, w, h, FrameToBytes(group, gi));
                group.Dispose();
                cancellationToken.ThrowIfCancellationRequested();
                await Task.Yield();
            }
        }
        finally
        {
            vaeLatent.Dispose();
        }
    }

    /// <summary>Runs the shared denoise loop and returns the VAE-ready latent <c>[1,48,gridT,vaeLatentH,vaeLatentW]</c>. The transformer weights are freed before return.</summary>
    private Tensor RunDenoise(int[] promptTokenIds, int[] negativeTokenIds, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress, out int seed)
    {
        ThrowIfDisposed();
        seed = request.Seed ?? SeedGenerator.RandomSeed();
        const int totalDownscale = 32;
        if (request.Width % totalDownscale != 0 || request.Height % totalDownscale != 0)
            throw new ArgumentException($"Width and height must be divisible by {totalDownscale} for Lance video.");
        if (numFrames < 1 || (numFrames - 1) % _config.VaeDownsampleTemporal != 0)
            throw new ArgumentException($"num_frames must satisfy (num_frames-1) % {_config.VaeDownsampleTemporal} == 0 (e.g. 1, 5, 9, … 121); got {numFrames}.");

        int vaeLatentH = request.Height / 16, vaeLatentW = request.Width / 16;
        int gridT = (numFrames - 1) / _config.VaeDownsampleTemporal + 1;
        int gridH = vaeLatentH / 2, gridW = vaeLatentW / 2;
        int nVae = gridT * gridH * gridW;
        int steps = request.Steps > 0 ? request.Steps : _config.NumTimesteps;
        float cfg = request.CfgScale > 0 ? request.CfgScale : _config.CfgTextScale;
        float shift = _config.VideoTimestepShift;

        Logs.Info($"Lance T2V denoise: {numFrames}f {request.Width}x{request.Height}, {steps} steps, cfg={cfg}, seed={seed} (grid {gridT}x{gridH}x{gridW}, {nVae} tokens)");
        Logs.Warning("Lance video pipeline is first-run-validation pending — numerics unverified vs the reference checkpoint.");

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        (Tensor condPos, int[] condUnd, int[] condGen) = LancePipelineCommon.BuildSequence(promptTokenIds.Length, gridT, gridH, gridW);
        (Tensor uncondPos, int[] uncondUnd, int[] uncondGen) = LancePipelineCommon.BuildSequence(negativeTokenIds.Length, gridT, gridH, gridW);

        Tensor latents = SeedGenerator.CreateNoise(new TensorShape(nVae, _config.PatchFeatureDim), seed);
        float[] tsteps = LancePipelineCommon.BuildShiftedTimesteps(steps, shift);

        for (int k = 0; k < steps; k++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = tsteps[k], tNext = tsteps[k + 1], dt = t - tNext;
            Tensor vCond = _transformer.Forward(Backend, promptTokenIds, latents, (gridT, gridH, gridW), t, condPos, condUnd, condGen, null);
            Tensor vUncond = _transformer.Forward(Backend, negativeTokenIds, latents, (gridT, gridH, gridW), t, uncondPos, uncondUnd, uncondGen, null);
            LancePipelineCommon.EulerCfgStep(latents, vCond, vUncond, cfg, dt);
            vCond.Dispose();
            vUncond.Dispose();
            stepSw.Stop();
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        condPos.Dispose(); uncondPos.Dispose();
        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        Tensor latentCl = LanceLatentPatch.Unpatchify(latents, gridT, gridH, gridW, 1, 2, 2, VaeChannels);
        latents.Dispose();
        Tensor vaeLatent = LancePipelineCommon.ChannelLastToBcthw(latentCl);   // [1,48,gridT,vaeLatentH,vaeLatentW]
        latentCl.Dispose();
        return vaeLatent;
    }

    /// <summary>Extracts frame <paramref name="frameIndex"/> from decoded RGB <c>[1,3,F,H,W]</c> in [-1,1] as interleaved-RGB bytes [0,255].</summary>
    private static byte[] FrameToBytes(Tensor rgb, int frameIndex)
    {
        int c = (int)rgb.Shape[1], f = (int)rgb.Shape[2], h = (int)rgb.Shape[3], w = (int)rgb.Shape[4];
        long frame = (long)h * w;
        byte[] outB = new byte[h * w * 3];
        float* p = (float*)rgb.DataPointer;
        for (long pix = 0; pix < frame; pix++)
            for (int ci = 0; ci < 3; ci++)
            {
                float v = ci < c ? p[((long)ci * f + frameIndex) * frame + pix] : 0f;
                int b = (int)MathF.Round((v + 1.0f) * 127.5f);
                outB[pix * 3 + ci] = (byte)Math.Clamp(b, 0, 255);
            }
        return outB;
    }
}
