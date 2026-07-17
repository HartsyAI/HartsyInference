using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Boogu-Image-0.1 pipeline (Base / Turbo text-to-image and Edit text+image-to-image). Caller supplies the
/// Qwen3-VL instruction embeddings (final hidden state <c>[1, T, 4096]</c>) — see
/// <c>docs/Research/BOOGU_IMAGE.md §5/§7</c> for how those are produced (text-only for T2I; via the Qwen3-VL vision
/// tower for edit). This mirrors the established <see cref="OmniGen2Pipeline"/> contract where the encoder forward is
/// owned outside the pipeline.
///
/// <para>T2I uses single CFG <c>pred = neg + tg·(cond − neg)</c>. Edit uses Boogu double guidance
/// <c>pred = cond + (tg − 1)·(cond − drop_text) + (ig − 1)·(drop_text − drop_all)</c> with three transformer passes per
/// step (cond = text+ref, drop_text = neg+ref, drop_all = neg+no-ref). Sampling is the ascending v1
/// flow-match schedule (<see cref="BooguFlowMatchScheduler"/>).</para></summary>
public sealed class BooguImagePipeline : DiffusionPipelineBase
{
    private readonly BooguImageTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly BooguImageConfig _config;

    /// <summary>Standard-profile DiT residency (HARTSY_KEEP_MODELS, default ON): the ~10 GB fp8 transformer
    /// stays GPU-resident across generations. The caller (loader) owns the Qwen3-VL TE staging — it must evict
    /// the resident DiT via <see cref="EvictResidentWeights"/> before an encode that needs the VRAM.</summary>
    private static readonly bool KeepModelsResident =
        EnvSwitch.IsEnabled("HARTSY_KEEP_MODELS", defaultOn: true);
    private bool _ditResident;

    /// <summary>Frees the resident transformer weights (no-op when not resident). For the loader's TE ⇄ DiT
    /// staging on a prompt-cache miss.</summary>
    public void EvictResidentWeights()
    {
        if (!_ditResident) return;
        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());
        _ditResident = false;
    }

    /// <summary>Creates a Boogu-Image pipeline. Pass <paramref name="vaeEncoder"/> to enable the edit path (it encodes
    /// reference images into the DiT latent stream); null restricts the pipeline to T2I. Caller owns each component.</summary>
    public BooguImagePipeline(IBackend backend, BooguImageTransformer transformer, VaeDecoder vaeDecoder,
        VaeEncoder? vaeEncoder, BooguImageConfig config)
        : base(backend)
    {
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
    }

    /// <summary>Text-to-image. <paramref name="instructionEmbeddings"/> is the Qwen3-VL final hidden state
    /// <c>[1, T, 4096]</c> for the (positive) instruction; <paramref name="negativeInstructionEmbeddings"/> is required
    /// when <paramref name="textGuidanceScale"/> &gt; 1 (use the empty-string encode). Turbo: pass
    /// <c>textGuidanceScale = 1</c> and ~4 steps.</summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor instructionEmbeddings,
        TextToImageRequest request,
        float textGuidanceScale = 4.0f,
        Tensor? negativeInstructionEmbeddings = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        if (textGuidanceScale > 1.0f && negativeInstructionEmbeddings is null)
            throw new ArgumentException("negativeInstructionEmbeddings is required when textGuidanceScale > 1.0.",
                nameof(negativeInstructionEmbeddings));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, _, int width, int height) = GenerationDefaults.Generic.Resolve(request);
        int latentH = height / 8;
        int latentW = width / 8;
        int seqLen = (latentH / _config.PatchSize) * (latentW / _config.PatchSize);

        Logs.Info($"Boogu-Image t2i: {width}x{height}, {steps} steps, tg={textGuidanceScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        TensorShape latentShape = new(1, _config.InChannels, latentH, latentW);
        int hPacked = latentH / _config.PatchSize;
        int wPacked = latentW / _config.PatchSize;

        // Packed-latent loop (the Z-Image/Krea2 token-space pattern): patchify the seeded noise ONCE, keep
        // the latent in [1, imgLen, p²·C] token space across the whole loop (ForwardPacked skips the
        // per-forward host PatchifyNCHW — a full D2H drain of the latent every forward), unpatchify ONCE
        // on-device before the VAE. Seed-compatible: noise is seeded in NCHW exactly as before.
        Tensor noiseNchw = SeedGenerator.CreateNoise(latentShape, seed);
        Tensor latent = DiTUtils.PatchifyNCHW(noiseNchw, _config.PatchSize);
        noiseNchw.Dispose();

        BooguFlowMatchScheduler scheduler = new(seqLen);
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        _ditResident = true;

        // Drain-free loop: CFG combine + Euler run as ONE in-place device op (CfgEulerStep:
        // z += (neg + gw·(pos−neg))·dt ≡ uncond + tg·(cond−uncond) Euler; gw=1 degenerates to the plain
        // step). The old host scheduler.Step + ApplyCfg forced a velocity D2H + latent re-upload per step.
        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor velocity = _transformer.ForwardPacked(Backend, latent, t, instructionEmbeddings, hPacked, wPacked);
            if (textGuidanceScale > 1.0f)
            {
                Tensor uncond = _transformer.ForwardPacked(Backend, latent, t, negativeInstructionEmbeddings!, hPacked, wPacked);
                Backend.CfgEulerStep(latent, velocity, uncond, textGuidanceScale, scheduler.Dt(i));
                uncond.Dispose();
            }
            else
            {
                Backend.CfgEulerStep(latent, velocity, velocity, 1.0f, scheduler.Dt(i));
            }
            velocity.Dispose();

            stepSw.Stop();
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        Backend.Sync();
        if (!KeepModelsResident)
        {
            Backend.FreeWeights(_transformer.EnumerateWeights());
            _ditResident = false;
        }

        // One device unpatchify back to NCHW for the VAE (the packed latent never left the GPU).
        Tensor unpacked = new Tensor(latentShape, DType.F32);
        Backend.UnpatchifyTokens(unpacked, latent, _config.InChannels, hPacked, wPacked, _config.PatchSize,
            innerChannelFastest: true);
        latent.Dispose();

        Backend.PreloadWeights(_vaeDecoder.EnumerateWeights());
        Tensor decoded = _vaeDecoder.Decode(Backend, unpacked);
        unpacked.Dispose();
        byte[] rgb = DeviceRgb(decoded);
        decoded.Dispose();
        Backend.FreeActivations();

        sw.Stop();
        Logs.Info($"Boogu-Image t2i complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (rgb, width, height, seed);
    }

    /// <summary>Device CHW F32 → HWC u8 conversion (one 3-byte/pixel D2H instead of the 4-byte/channel host
    /// loop; see Krea2/Z-Image).</summary>
    private byte[] DeviceRgb(Tensor image)
    {
        int outH = (int)image.Shape[2], outW = (int)image.Shape[3];
        Tensor hwcU8 = new Tensor(new TensorShape(outH, outW, 3), DType.U8);
        Backend.ChwF32ToHwcU8(hwcU8, image);
        byte[] rgb = new byte[(long)outH * outW * 3];
        unsafe
        {
            fixed (byte* dst = rgb)
                Buffer.MemoryCopy((void*)hwcU8.DataPointer, dst, rgb.Length, rgb.Length);
        }
        hwcU8.Dispose();
        return rgb;
    }

    /// <summary>Edit (text+image-to-image) with Boogu double guidance. The three instruction embeddings are produced by
    /// the caller from the Qwen3-VL encoder: <paramref name="condEmbeddings"/> (positive instruction with the reference
    /// image seen by the vision tower), <paramref name="dropTextEmbeddings"/> (negative instruction, image still seen)
    /// and <paramref name="dropAllEmbeddings"/> (negative instruction, no image). Reference images are RGB tensors
    /// <c>[1, 3, Hr, Wr]</c> in <c>[-1, 1]</c>; they are VAE-encoded into the DiT latent stream. When
    /// <paramref name="imageGuidanceScale"/> ≤ 1 the drop-all pass is skipped (text-only guidance, reference kept).</summary>
    public (byte[] rgbData, int width, int height, int seed) EditFromEmbeddings(
        Tensor condEmbeddings,
        Tensor dropTextEmbeddings,
        Tensor? dropAllEmbeddings,
        IReadOnlyList<Tensor> referenceImages,
        TextToImageRequest request,
        float textGuidanceScale = 4.0f,
        float imageGuidanceScale = 1.0f,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        if (_vaeEncoder is null)
            throw new InvalidOperationException("Edit requires a VAE encoder; construct the pipeline with vaeEncoder != null.");
        if (referenceImages.Count == 0)
            throw new ArgumentException("Edit requires at least one reference image.", nameof(referenceImages));
        bool doubleGuide = imageGuidanceScale > 1.0f;
        if (doubleGuide && dropAllEmbeddings is null)
            throw new ArgumentException("dropAllEmbeddings is required when imageGuidanceScale > 1.0.", nameof(dropAllEmbeddings));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, _, int width, int height) = GenerationDefaults.Generic.Resolve(request);
        int latentH = height / 8;
        int latentW = width / 8;
        int seqLen = (latentH / _config.PatchSize) * (latentW / _config.PatchSize);

        Logs.Info($"Boogu-Image edit: {width}x{height}, {steps} steps, tg={textGuidanceScale}, ig={imageGuidanceScale}, refs={referenceImages.Count}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // Encode reference images into latents once.
        Tensor[] refLatents = new Tensor[referenceImages.Count];
        for (int j = 0; j < referenceImages.Count; j++)
            refLatents[j] = _vaeEncoder.Encode(Backend, referenceImages[j]);

        TensorShape latentShape = new(1, _config.InChannels, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);

        BooguFlowMatchScheduler scheduler = new(seqLen);
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        _ditResident = true;

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor cond = _transformer.Forward(Backend, latent, t, condEmbeddings, refLatents);
            Tensor dropText = _transformer.Forward(Backend, latent, t, dropTextEmbeddings, refLatents);

            if (doubleGuide)
            {
                // 3-tensor double guidance — no 2-tensor device fusion; keep the host combine + step.
                // Boogu's cond + (tg−1)(cond − dropText) + (ig−1)(dropText − dropAll) is algebraically the shared
                // dual-CFG form uncond + ig·(mid − uncond) + tg·(cond − mid) — see CfgHelper.ApplyDualCfg.
                Tensor dropAll = _transformer.Forward(Backend, latent, t, dropAllEmbeddings!, null);
                Tensor velocity = CfgHelper.ApplyDualCfg(cond, dropText, dropAll, textGuidanceScale, imageGuidanceScale);
                dropAll.Dispose();
                Tensor newLatent = new(latentShape, DType.F32);
                scheduler.Step(newLatent, velocity, latent, i);
                velocity.Dispose();
                latent.Dispose();
                latent = newLatent;
            }
            else
            {
                // Text-only guidance with the reference kept: cond + (tg-1)*(cond - dropText), fused with the
                // Euler update as one in-place device op (≡ dropText + tg·(cond−dropText)).
                Backend.CfgEulerStep(latent, cond, dropText, textGuidanceScale, scheduler.Dt(i));
            }
            cond.Dispose();
            dropText.Dispose();

            stepSw.Stop();
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        Backend.Sync();
        if (!KeepModelsResident)
        {
            Backend.FreeWeights(_transformer.EnumerateWeights());
            _ditResident = false;
        }
        foreach (Tensor r in refLatents) r.Dispose();

        Backend.PreloadWeights(_vaeDecoder.EnumerateWeights());
        Tensor decoded = _vaeDecoder.Decode(Backend, latent);
        latent.Dispose();
        byte[] rgb = DeviceRgb(decoded);
        decoded.Dispose();
        Backend.FreeActivations();

        sw.Stop();
        Logs.Info($"Boogu-Image edit complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (rgb, width, height, seed);
    }

}
