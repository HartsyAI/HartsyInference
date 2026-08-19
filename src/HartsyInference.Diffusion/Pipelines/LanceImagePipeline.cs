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

/// <summary>Lance (ByteDance, Apache-2.0) text-to-image pipeline. Wires <see cref="LanceTransformer"/> (MoT backbone), the packed-sequence builders in <see cref="LancePipelineCommon"/>, and <see cref="Wan22VaeDecoder"/> into the T2I path from upstream <c>validation_gen</c> (reconciled against the real <c>Lance_3B</c> checkpoint).
///
/// <para>Per upstream inference defaults: 2-way text CFG gated to shifted time in <c>(0.4, 1]</c> with global renorm; the uncond branch is the same chat-templated sequence with the caption removed (or the negative prompt substituted when provided). Vision CFG / editing (the frozen Qwen2.5-VL ViT) is not part of this release's T2I path.</para>
///
/// <para>Spatial math: the VAE downsamples 16× and the transformer patchifies (1,1,1), so an H×W image gives a <c>[48, 1, H/16, W/16]</c> latent = <c>(1, H/16, W/16)</c> token grid of 48-dim tokens. H and W must be divisible by 16, and H/16, W/16 ≤ 64 (the frozen position table).</para></summary>
public sealed unsafe class LanceImagePipeline : DiffusionPipelineBase
{
    private const int VaeChannels = 48;

    private readonly LanceTransformer _transformer;
    private readonly Wan22VaeDecoder _vae;
    private readonly Wan22VaeEncoder? _vaeEncoder;
    private readonly LanceConfig _config;
    private readonly LancePromptTemplate _template;

    /// <summary>Creates a Lance T2I pipeline. Img2img is unavailable; use the overload accepting a <see cref="Wan22VaeEncoder"/> to enable it.</summary>
    public LanceImagePipeline(IBackend backend, LanceTransformer transformer, Wan22VaeDecoder vae, LanceConfig config,
        LancePromptTemplate? promptTemplate = null)
        : this(backend, transformer, vae, vaeEncoder: null, config, promptTemplate)
    {
    }

    /// <summary>Creates a Lance pipeline with both VAE halves loaded — required for img2img (pass an <see cref="ImageToImageRequest"/> to <see cref="GenerateFromTokens"/>). Masked inpaint is NOT supported. <paramref name="promptTemplate"/> should be the chat-templated scaffold from <see cref="LancePromptTemplate.Create"/>; null falls back to the bare (<c>text_template=false</c>) layout.</summary>
    public LanceImagePipeline(IBackend backend, LanceTransformer transformer, Wan22VaeDecoder vae,
        Wan22VaeEncoder? vaeEncoder, LanceConfig config, LancePromptTemplate? promptTemplate = null)
        : base(backend)
    {
        _transformer = transformer;
        _vae = vae;
        _vaeEncoder = vaeEncoder;
        _config = config;
        _template = promptTemplate ?? LancePromptTemplate.Bare(config);
    }

    /// <summary>Generates an image from raw caption token ids (plain Qwen2 BPE — the chat scaffold is added internally from the prompt template). <paramref name="negativeTokenIds"/> empty replicates the upstream uncond (caption dropped); non-empty substitutes it as the uncond caption.
    /// <para>An <see cref="ImageToImageRequest"/> selects img2img: the source is encoded through the Wan2.2 VAE
    /// (normalized latent), flattened to 48-dim tokens and mixed with fresh noise at <c>t = tsteps[startStep]</c>
    /// (<c>x = (1−t)·src + t·noise</c>) — requires a <see cref="Wan22VaeEncoder"/> on construction. Masked
    /// inpaint is not supported (throws <see cref="NotSupportedException"/>): the 16× downscale leaves one mask
    /// cell per 16×16-pixel block, too coarse for blend-on-vanilla. Strength=0 short-circuits to byte-identical
    /// pass-through.</para></summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds, int[] negativeTokenIds, TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        // Wrap-pad every conv backend for this call so the output tiles seamlessly; restores on dispose.
        using IDisposable seamlessScope = BeginSeamlessTiling(request.SeamlessTiling);
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a Wan22VaeEncoder. Construct the pipeline with the overload that accepts one.");
        // Masked inpaint runs at token granularity — one mask cell per 16x16-pixel block from the 16x VAE
        // downscale. Coarser than Flux's 8x, but the final pixel recomposite makes everything outside the mask
        // byte-exact source, so the coarseness only widens the transition band at the mask boundary.

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? GenerationDefaults.Generic.Width;
        int height = request.Height ?? GenerationDefaults.Generic.Height;
        int downscale = _config.VaeDownsampleSpatial * _config.LatentPatchSize.H;
        if (width % downscale != 0 || height % downscale != 0)
            throw new ArgumentException($"Width and height must be divisible by {downscale} for Lance.");

        int gridT = 1, gridH = height / downscale, gridW = width / downscale;
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
        Stopwatch sw = Stopwatch.StartNew();

        Stopwatch preloadSw = Stopwatch.StartNew();
        Backend.PreloadWeights(_transformer.EnumerateWeights());
        Logs.Info($"Transformer preload done in {preloadSw.ElapsedMilliseconds}ms");

        using LanceSequence cond = LancePipelineCommon.BuildGenSequence(_template, promptTokenIds, _config, gridT, gridH, gridW);
        using LanceSequence uncond = LancePipelineCommon.BuildGenSequence(_template, negativeTokenIds, _config, gridT, gridH, gridW);
        int[] latentPosIds = LancePipelineCommon.BuildLatentPositionIds(gridT, gridH, gridW, _config.MaxLatentSize);

        float[] tsteps = LancePipelineCommon.BuildShiftedTimesteps(steps, shift);
        Tensor latents = BuildInitialTokenLatents(request, tsteps, nVae, seed, startStep, out Tensor? sourceTokensKeep);
        Tensor? maskPixel = plan.MaskPixel;
        bool isMaskedInpaint = maskPixel is not null && sourceTokensKeep is not null;
        // Token grid is row-major (t, h, w) with T=1, so the mask downsampled to gridHxgridW lines up 1:1.
        Tensor? tokenMask = isMaskedInpaint
            ? MaskBlendUtilities.DownsampleMaskAreaAverage(maskPixel!, gridH, gridW)
            : null;

        for (int k = startStep; k < steps; k++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = tsteps[k];
            float dt = t - tsteps[k + 1];

            {
                using Tensor vCond = _transformer.Forward(Backend, cond.TextTokenIds, latents, latentPosIds, t,
                    cond.PositionIds, cond.UndIdx, cond.GenIdx, cond.AttentionMask);
                // Upstream cfg_interval=[0.4,1]: late low-noise steps run cond-only (skips the uncond forward entirely).
                if (cfg > 1f && t > _config.CfgIntervalMin)
                {
                    using Tensor vUncond = _transformer.Forward(Backend, uncond.TextTokenIds, latents, latentPosIds, t,
                        uncond.PositionIds, uncond.UndIdx, uncond.GenIdx, uncond.AttentionMask);
                    Backend.CfgRenormEulerStep(latents, vCond, vUncond, cfg, -dt, _config.CfgRenormMin);
                }
                else
                {
                    Backend.CfgEulerStep(latents, vCond, vCond, 1f, -dt);
                }
            }

            // Masked-inpaint blend: unmasked tokens back on the source's flow trajectory at the NEXT step's
            // sigma (tsteps has steps+1 entries and tsteps[steps] = 0, so the final step blends clean source).
            if (isMaskedInpaint)
            {
                Tensor fresh = SeedGenerator.CreateNoise(latents.Shape, seed + k + 1);
                Tensor noisedSource = new Tensor(latents.Shape, DType.F32);
                Img2ImgSetup.MixAtSigma(noisedSource, sourceTokensKeep!, fresh, tsteps[k + 1]);
                fresh.Dispose();
                MaskBlendUtilities.BlendTokensInPlace(latents, noisedSource, tokenMask!);
                noisedSource.Dispose();
            }
            stepSw.Stop();
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        // tokens → channel-last latent → [1,48,1,Hlat,Wlat] → VAE decode.
        (int pt, int ph, int pw) = _config.LatentPatchSize;
        Tensor latentCl = LanceLatentPatch.Unpatchify(latents, gridT, gridH, gridW, pt, ph, pw, VaeChannels);
        latents.Dispose();
        Tensor vaeLatent = LancePipelineCommon.ChannelLastToBcthw(latentCl);   // [1,48,1,H/16,W/16]
        latentCl.Dispose();

        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor rgb = _vae.Decode(Backend, vaeLatent);       // [1,3,1,H,W]
        vaeLatent.Dispose();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // Pixel recomposite: rgb is [1,3,1,H,W]; a zero-copy [1,3,H,W] view lets the shared 4D blend run.
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            Tensor rgb4d = rgb.Reshape(new TensorShape([1L, 3, rgb.Shape[3], rgb.Shape[4]]));
            MaskBlendUtilities.BlendChannelsInPlace(rgb4d, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }
        sourceTokensKeep?.Dispose();
        tokenMask?.Dispose();

        byte[] bytes = Rgb5dToBytes(rgb);
        rgb.Dispose();

        sw.Stop();
        Logs.Info($"Lance T2I complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (bytes, width, height, seed);
    }

    /// <summary>Builds the initial token-space latents <c>[nVae, 48]</c>. T2I: pure seeded noise. Img2img: the
    /// source image goes <c>Wan22VaeEncoder.EncodeFrame</c> (<c>[1,48,1,H/16,W/16]</c>, normalized latent space —
    /// the same space the T2I loop denoises and the decoder consumes) → channel-last → token flatten →
    /// <c>Img2ImgSetup.MixAtSigma</c> with the fresh noise at <c>t = tsteps[startStep]</c>.</summary>
    private Tensor BuildInitialTokenLatents(TextToImageRequest request, float[] tsteps,
        int nVae, int seed, int startStep, out Tensor? sourceTokensKeep)
    {
        sourceTokensKeep = null;
        Tensor noise = TakeOrCreateNoise(request, new TensorShape(nVae, _config.PatchFeatureDim), seed);
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
        (int pt, int ph, int pw) = _config.LatentPatchSize;
        Tensor sourceTokens = LanceLatentPatch.Patchify(channelLast, pt, ph, pw);  // [nVae, 48]
        channelLast.Dispose();
        if ((int)sourceTokens.Shape[0] != nVae)
            throw new InvalidOperationException($"Encoded source produced {sourceTokens.Shape[0]} tokens, expected {nVae}.");

        Tensor latents = new Tensor(new TensorShape(nVae, _config.PatchFeatureDim), DType.F32);
        Img2ImgSetup.MixAtSigma(latents, sourceTokens, noise, tsteps[startStep]);
        if (img2img.Mask is not null)
        {
            sourceTokensKeep = sourceTokens;
        }
        else
        {
            sourceTokens.Dispose();
        }
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
