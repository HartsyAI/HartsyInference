using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Krea 2 text-to-image pipeline. Encodes the prompt through Qwen3-VL-4B (12-layer hidden-state tap) via
/// <see cref="LlamaStyleEncoder.EncodeMultiLayer"/>, runs the <see cref="Krea2Transformer"/> (which fuses the tapped
/// layers, patchifies the latent, and predicts the flow-match velocity) under a resolution-aware flow-match Euler
/// schedule, and decodes through the 16-channel Qwen-Image VAE. CFG is a dual pass when <c>request.CfgScale &gt; 1</c>
/// (Base, ~4.5); Turbo/TDM runs a single conditional pass (guidance off). See <c>docs/Research/KREA2.md</c>.</summary>
public sealed class Krea2Pipeline : DiffusionPipelineBase
{
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly Krea2Transformer _transformer;
    private readonly QwenImageVaeDecoder _vaeDecoder;
    private readonly Krea2Config _config;

    /// <summary>Creates a Krea 2 pipeline. The caller owns each component's lifetime (they may be shared/reused).</summary>
    public Krea2Pipeline(IBackend backend, LlamaStyleEncoder textEncoder, Krea2Transformer transformer,
        QwenImageVaeDecoder vaeDecoder, Krea2Config config)
        : base(backend)
    {
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image. <paramref name="promptTokenIds"/> / <paramref name="negativeTokenIds"/> are the
    /// chat-templated Qwen token sequences; the leading <paramref name="promptDropIndex"/> (system-prefix) hidden
    /// states are dropped (Krea 2's <c>prompt_template_encode_start_idx = 34</c>). The negative stream is used only
    /// when <c>request.CfgScale &gt; 1</c> (Base); for Turbo pass <c>CfgScale ≤ 1</c>.</summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds,
        int[]? negativeTokenIds,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null,
        int promptDropIndex = 34,
        int negativeDropIndex = 34)
    {
        ThrowIfDisposed();

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width;
        int height = request.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int hPacked = latentH / _config.PatchSize;
        int wPacked = latentW / _config.PatchSize;
        int imageSeqLen = hPacked * wPacked;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;
        bool useCfg = cfgScale > 1.0f && negativeTokenIds is not null;

        Logs.Info($"Krea 2 t2i: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}, distilled={_config.IsDistilled}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── encode prompt: 12-layer tap → [1, S, 12·2560], drop the system prefix ──
        Backend.PreloadWeights(_textEncoder.EnumerateWeights());
        Tensor condHidden = EncodeTapped(promptTokenIds, promptDropIndex);
        Tensor? uncondHidden = useCfg ? EncodeTapped(negativeTokenIds!, negativeDropIndex) : null;
        Backend.FreeWeights(_textEncoder.EnumerateWeights());

        // ── scheduler: resolution-aware exp shift (Turbo pins mu=1.15) ──
        FlowMatchEulerDiscreteScheduler scheduler = _config.IsDistilled
            ? new FlowMatchEulerDiscreteScheduler(MathF.Exp(1.15f))
            : FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(imageSeqLen, baseSeqLen: 256, maxSeqLen: 6400,
                baseShift: 0.5f, maxShift: 1.15f);
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        TensorShape latentShape = new(1, _config.VaeChannels, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(latentShape, DType.F32);
            Backend.Scale(scaled, latent, initSigma);
            latent.Dispose();
            latent = scaled;
        }

        Backend.PreloadWeights(_transformer.EnumerateWeights());

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i] / 1000.0f; // scheduler stores sigma·1000; transformer takes t∈[0,1]

            Tensor noisePred;
            if (useCfg)
            {
                Tensor cond = _transformer.Forward(Backend, latent, t, condHidden);
                Tensor uncond = _transformer.Forward(Backend, latent, t, uncondHidden!);
                noisePred = CfgHelper.ApplyCfg(uncond, cond, cfgScale);
                uncond.Dispose();
                cond.Dispose();
            }
            else
            {
                noisePred = _transformer.Forward(Backend, latent, t, condHidden);
            }

            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePred, latent, i);
            noisePred.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        condHidden.Dispose();
        uncondHidden?.Dispose();

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        Backend.PreloadWeights(_vaeDecoder.EnumerateWeights());
        Tensor image = _vaeDecoder.Decode(Backend, latent);
        latent.Dispose();
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Krea 2 t2i complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (rgbData, width, height, seed);
    }

    /// <summary>Encodes a token sequence, stacks the 12 selected layers (tap-major <c>[1, S, 12·2560]</c>) and drops
    /// the first <paramref name="dropIndex"/> token positions (the chat-template system prefix).</summary>
    private unsafe Tensor EncodeTapped(int[] tokenIds, int dropIndex)
    {
        Tensor full = _textEncoder.EncodeMultiLayer(Backend, [tokenIds], Krea2Config.TextEncoderSelectLayers,
            interleavedLayout: false);
        if (dropIndex <= 0) return full;

        int s = (int)full.Shape[1];
        int feat = (int)full.Shape[2];
        int keep = Math.Max(1, s - dropIndex);
        Tensor sliced = new Tensor(new TensorShape(1, keep, feat), DType.F32);
        long rowBytes = (long)feat * sizeof(float);
        float* src = (float*)full.DataPointer + (long)dropIndex * feat;
        Buffer.MemoryCopy(src, (void*)sliced.DataPointer, (long)keep * rowBytes, (long)keep * rowBytes);
        full.Dispose();
        return sliced;
    }
}
