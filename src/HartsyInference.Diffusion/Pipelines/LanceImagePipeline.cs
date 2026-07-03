using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Lance (ByteDance, Apache-2.0) text-to-image pipeline. Wires the verified components — <see cref="LanceTransformer"/> (MoT backbone), <see cref="LanceLatentPatch"/> (latent↔token handoff), <see cref="Wan22VaeDecoder"/> — into the T2I path distilled from upstream <c>lance.py</c> <c>validation_gen</c>.
///
/// <para><b>Status: built, first-run validation pending.</b> The component blocks are individually unit-tested, but the sequence packing, the MaPE position offsets, and the CFG combine are numerically validation-gated against the real Lance checkpoint (no weights in this environment). Debug-dump hooks (<c>LANCE_DEBUG_DIR</c>) are wired in the transformer for the layer-diff.</para>
///
/// <para>T2I uses standard 2-way text CFG (cond prompt vs uncond/negative prompt). The 3-way vision CFG and image-editing path require the frozen Qwen2.5-VL ViT (deferred). Video (T&gt;1) requires the VAE streaming decode (Phase 9).</para>
///
/// <para>Spatial math: VAE downsamples 16×, the transformer patchifies (1,2,2), so for an H×W image the VAE latent is <c>[48, 1, H/16, W/16]</c> and the transformer token grid is <c>(1, H/32, W/32)</c> with 192-dim tokens. H and W must be divisible by 32.</para></summary>
public sealed unsafe class LanceImagePipeline : DiffusionPipelineBase
{
    private const int VaeChannels = 48;

    private readonly LanceTransformer _transformer;
    private readonly Wan22VaeDecoder _vae;
    private readonly Wan22VaeEncoder? _vaeEncoder;
    private readonly LanceConfig _config;

    /// <summary>Creates a Lance T2I pipeline. Img2img is unavailable; use the overload accepting a <see cref="Wan22VaeEncoder"/> to enable it.</summary>
    public LanceImagePipeline(IBackend backend, LanceTransformer transformer, Wan22VaeDecoder vae, LanceConfig config)
        : this(backend, transformer, vae, vaeEncoder: null, config)
    {
    }

    /// <summary>Creates a Lance pipeline with both VAE halves loaded — required for img2img (pass an <see cref="ImageToImageRequest"/> to <see cref="GenerateFromTokens"/>). Masked inpaint is NOT supported (see <see cref="GenerateFromTokens"/>).</summary>
    public LanceImagePipeline(IBackend backend, LanceTransformer transformer, Wan22VaeDecoder vae,
        Wan22VaeEncoder? vaeEncoder, LanceConfig config)
        : base(backend)
    {
        _transformer = transformer;
        _vae = vae;
        _vaeEncoder = vaeEncoder;
        _config = config;
    }

    /// <summary>Generates an image from chat-templated prompt + negative-prompt token ids (Qwen2 BPE). Steps/CFG from <paramref name="request"/>.
    /// <para>An <see cref="ImageToImageRequest"/> selects img2img: the source is encoded through the Wan2.2 VAE
    /// (normalized latent), converted to Lance's 192-dim token space (channel-last + (1,2,2) patchify) and mixed with
    /// fresh noise at <c>t = tsteps[startStep]</c> (<c>x = (1−t)·src + t·noise</c>, matching the Euler loop's
    /// <c>x_t = (1−t)·x0 + t·ε</c> convention) — requires a <see cref="Wan22VaeEncoder"/> on construction. Masked
    /// inpaint is not supported (throws <see cref="NotSupportedException"/>): the 32× effective downscale leaves one
    /// mask cell per 32×32-pixel block, too coarse for blend-on-vanilla. Strength=0 short-circuits to byte-identical
    /// pass-through.</para></summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds, int[] negativeTokenIds, TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a Wan22VaeEncoder. Construct the pipeline with the overload that accepts one.");
        if (isImg2Img && ((ImageToImageRequest)request).Mask is not null)
            throw new NotSupportedException("Lance masked inpaint is not supported: the 32× effective downscale (VAE 16× × patch 2×) leaves one mask cell per 32×32-pixel block, too coarse for blend-on-vanilla.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? GenerationDefaults.Generic.Width;
        int height = request.Height ?? GenerationDefaults.Generic.Height;
        const int totalDownscale = 32;   // VAE 16× × transformer patch 2×
        if (width % totalDownscale != 0 || height % totalDownscale != 0)
            throw new ArgumentException($"Width and height must be divisible by {totalDownscale} for Lance.");

        int vaeLatentH = height / 16, vaeLatentW = width / 16;
        int gridT = 1, gridH = vaeLatentH / 2, gridW = vaeLatentW / 2;
        int nVae = gridT * gridH * gridW;
        int steps = request.Steps ?? _config.NumTimesteps;
        float cfg = request.CfgScale ?? _config.CfgTextScale;
        float shift = _config.ImageTimestepShift;

        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(request, height, width, steps);
        if (plan.PassThrough)
        {
            Logs.Info("Strength=0; passing source through unchanged");
            return (ImagePostProcessor.TensorToRgbBytes(((ImageToImageRequest)request).SourceImage), width, height, seed);
        }
        int startStep = plan.StartStep;

        string opMode = isImg2Img ? $"img2img (start={startStep}/{steps})" : "T2I";
        Logs.Info($"Lance {opMode}: {width}x{height}, {steps} steps, cfg={cfg}, seed={seed} (grid {gridT}x{gridH}x{gridW}, {nVae} tokens)");
        Logs.Warning("Lance pipeline is first-run-validation pending — numerics unverified vs the reference checkpoint.");
        Stopwatch sw = Stopwatch.StartNew();

        Backend.PreloadWeights(_transformer.EnumerateWeights());

        // Build cond / uncond sequence metadata.
        (Tensor condPos, int[] condUnd, int[] condGen) = LancePipelineCommon.BuildSequence(promptTokenIds.Length, gridT, gridH, gridW);
        (Tensor uncondPos, int[] uncondUnd, int[] uncondGen) = LancePipelineCommon.BuildSequence(negativeTokenIds.Length, gridT, gridH, gridW);

        // Logit-normal-shifted timestep grid (t: 1 → 0).
        float[] tsteps = LancePipelineCommon.BuildShiftedTimesteps(steps, shift);

        // Initial 192-dim token latents (t2i: pure noise; img2img: encoded source mixed with noise at tsteps[startStep]).
        Tensor latents = BuildInitialTokenLatents(request, tsteps, nVae, seed, startStep);

        for (int k = startStep; k < steps; k++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = tsteps[k];
            float tNext = tsteps[k + 1];
            float dt = t - tNext;

            Tensor vCond = _transformer.Forward(Backend, promptTokenIds, latents, (gridT, gridH, gridW), t, condPos, condUnd, condGen, null);
            Tensor vUncond = _transformer.Forward(Backend, negativeTokenIds, latents, (gridT, gridH, gridW), t, uncondPos, uncondUnd, uncondGen, null);

            // 2-way text CFG: v = uncond + cfg·(cond − uncond); Euler: z -= v·dt.
            LancePipelineCommon.EulerCfgStep(latents, vCond, vUncond, cfg, dt);
            vCond.Dispose();
            vUncond.Dispose();

            stepSw.Stop();
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        condPos.Dispose(); uncondPos.Dispose();

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        // tokens → channel-last latent → [1,48,1,Hlat,Wlat] → VAE decode.
        Tensor latentCl = LanceLatentPatch.Unpatchify(latents, gridT, gridH, gridW, 1, 2, 2, VaeChannels);
        latents.Dispose();
        Tensor vaeLatent = LancePipelineCommon.ChannelLastToBcthw(latentCl);   // [1,48,1,vaeLatentH,vaeLatentW]
        latentCl.Dispose();

        Tensor rgb = _vae.Decode(Backend, vaeLatent);       // [1,3,1,H,W]
        vaeLatent.Dispose();

        byte[] bytes = Rgb5dToBytes(rgb);
        rgb.Dispose();

        sw.Stop();
        Logs.Info($"Lance T2I complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (bytes, width, height, seed);
    }

    /// <summary>Builds the initial token-space latents <c>[nVae, 192]</c>. T2I: pure seeded noise. Img2img: the
    /// source image goes <c>Wan22VaeEncoder.EncodeFrame</c> (<c>[1,48,1,H/16,W/16]</c>, normalized latent space —
    /// the same space the T2I loop denoises and the decoder consumes) → channel-last → <c>(1,2,2)</c> patchify →
    /// <c>Img2ImgSetup.MixAtSigma</c> with the fresh noise at <c>t = tsteps[startStep]</c>.</summary>
    private Tensor BuildInitialTokenLatents(TextToImageRequest request, float[] tsteps,
        int nVae, int seed, int startStep)
    {
        Tensor noise = SeedGenerator.CreateNoise(new TensorShape(nVae, _config.PatchFeatureDim), seed);
        if (request is not ImageToImageRequest img2img) return noise;

        Stopwatch vaeEncSw = Stopwatch.StartNew();
        // The source arrives [1, 3, H, W]; the 3D-causal VAE wants a single-frame clip [1, 3, 1, H, W].
        // Reshape is a zero-copy view over the caller's pixel buffer (batch dim splits into batch+time).
        Tensor source5d = img2img.SourceImage.Reshape(new TensorShape(
            [1L, 3, 1, img2img.SourceImage.Shape[2], img2img.SourceImage.Shape[3]]));
        Tensor encoded = _vaeEncoder!.EncodeFrame(Backend, source5d);   // [1, 48, 1, H/16, W/16]
        vaeEncSw.Stop();
        Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

        Tensor channelLast = LancePipelineCommon.BcthwToChannelLast(encoded);   // [1, H/16, W/16, 48]
        encoded.Dispose();
        Tensor sourceTokens = LanceLatentPatch.Patchify(channelLast, 1, 2, 2);  // [nVae, 192]
        channelLast.Dispose();
        if ((int)sourceTokens.Shape[0] != nVae)
            throw new InvalidOperationException($"Encoded source produced {sourceTokens.Shape[0]} tokens, expected {nVae}.");

        Tensor latents = new Tensor(new TensorShape(nVae, _config.PatchFeatureDim), DType.F32);
        Img2ImgSetup.MixAtSigma(latents, sourceTokens, noise, tsteps[startStep]);
        sourceTokens.Dispose();
        noise.Dispose();
        return latents;
    }

    /// <summary>Converts decoded RGB <c>[1,3,1,H,W]</c> in [-1,1] to interleaved RGB bytes [0,255].</summary>
    private static byte[] Rgb5dToBytes(Tensor rgb)
    {
        int c = (int)rgb.Shape[1], h = (int)rgb.Shape[3], w = (int)rgb.Shape[4];
        byte[] outB = new byte[h * w * 3];
        float* p = (float*)rgb.DataPointer;
        long frame = (long)h * w;
        for (int hi = 0; hi < h; hi++)
            for (int wi = 0; wi < w; wi++)
            {
                long pix = (long)hi * w + wi;
                for (int ci = 0; ci < 3; ci++)
                {
                    float v = ci < c ? p[(long)ci * frame + pix] : 0f;
                    int b = (int)MathF.Round((v + 1.0f) * 127.5f);
                    outB[pix * 3 + ci] = (byte)Math.Clamp(b, 0, 255);
                }
            }
        return outB;
    }
}
