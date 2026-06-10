using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Pipelines;

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
    private readonly LanceConfig _config;

    public LanceImagePipeline(IBackend backend, LanceTransformer transformer, Wan22VaeDecoder vae, LanceConfig config)
        : base(backend)
    {
        _transformer = transformer;
        _vae = vae;
        _config = config;
    }

    /// <summary>Generates an image from chat-templated prompt + negative-prompt token ids (Qwen2 BPE). Steps/CFG from <paramref name="request"/>.</summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds, int[] negativeTokenIds, TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        const int totalDownscale = 32;   // VAE 16× × transformer patch 2×
        if (request.Width % totalDownscale != 0 || request.Height % totalDownscale != 0)
            throw new ArgumentException($"Width and height must be divisible by {totalDownscale} for Lance.");

        int vaeLatentH = request.Height / 16, vaeLatentW = request.Width / 16;
        int gridT = 1, gridH = vaeLatentH / 2, gridW = vaeLatentW / 2;
        int nVae = gridT * gridH * gridW;
        int steps = request.Steps > 0 ? request.Steps : _config.NumTimesteps;
        float cfg = request.CfgScale > 0 ? request.CfgScale : _config.CfgTextScale;
        float shift = _config.ImageTimestepShift;

        Logs.Info($"Lance T2I: {request.Width}x{request.Height}, {steps} steps, cfg={cfg}, seed={seed} (grid {gridT}x{gridH}x{gridW}, {nVae} tokens)");
        Logs.Warning("Lance pipeline is first-run-validation pending — numerics unverified vs the reference checkpoint.");
        Stopwatch sw = Stopwatch.StartNew();

        Backend.PreloadWeights(_transformer.EnumerateWeights());

        // Build cond / uncond sequence metadata.
        (Tensor condPos, int[] condUnd, int[] condGen) = LancePipelineCommon.BuildSequence(promptTokenIds.Length, gridT, gridH, gridW);
        (Tensor uncondPos, int[] uncondUnd, int[] uncondGen) = LancePipelineCommon.BuildSequence(negativeTokenIds.Length, gridT, gridH, gridW);

        // Initial noise in the 192-dim token space.
        Tensor latents = SeedGenerator.CreateNoise(new TensorShape(nVae, _config.PatchFeatureDim), seed);

        // Logit-normal-shifted timestep grid (t: 1 → 0).
        float[] tsteps = LancePipelineCommon.BuildShiftedTimesteps(steps, shift);

        for (int k = 0; k < steps; k++)
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
        return (bytes, request.Width, request.Height, seed);
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
