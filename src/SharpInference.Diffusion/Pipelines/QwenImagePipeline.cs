using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>Qwen-Image text-to-image pipeline. Encodes the prompt through Qwen2.5-VL via <see cref="LlamaStyleEncoder"/>, packs the noisy latent into 2×2 patch tokens, runs <see cref="QwenImageTransformer"/> with flow-match Euler scheduling (dynamic shift), unpacks the predicted velocity back to <c>[B, 16, H, W]</c>, and decodes through the 16-channel Qwen-Image VAE. CFG is applied as a dual-pass when <c>request.CfgScale &gt; 1</c>; otherwise a single conditional forward is used. Transformer + text encoder weights are evicted from VRAM before VAE decode (Phase 3 deviations #18 / #33) so a 30 GB Qwen-Image FP8 stack still leaves room for tiled decode workspace on a 40 GB card.</summary>
public sealed unsafe class QwenImagePipeline : DiffusionPipelineBase
{
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly QwenImageTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly QwenImageConfig _config;

    /// <summary>Creates a Qwen-Image pipeline. Caller is responsible for the lifetime of the components — they may be reused across pipelines.</summary>
    public QwenImagePipeline(IBackend backend, LlamaStyleEncoder textEncoder,
        QwenImageTransformer transformer, VaeDecoder vaeDecoder, QwenImageConfig config)
        : base(backend)
    {
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-tokenized prompt input. The negative prompt tokens are ignored when <see cref="TextToImageRequest.CfgScale"/> ≤ 1.</summary>
    /// <param name="promptTokenIds">Conditional prompt token IDs (Qwen2.5-VL vocab).</param>
    /// <param name="negativeTokenIds">Negative-prompt token IDs (same length as <paramref name="promptTokenIds"/> recommended). Used only when <c>CfgScale &gt; 1</c>.</param>
    /// <param name="request">Generation parameters.</param>
    /// <param name="onProgress">Optional per-step progress callback.</param>
    /// <param name="promptDropIndex">Number of leading hidden-state positions to drop from the conditional
    /// stream after encoding. Qwen-Image wraps the prompt in a system+user-header template and discards
    /// those prefix tokens' hidden states (diffusers' <c>prompt_template_encode_start_idx</c>). 0 = keep all
    /// (raw-prompt path).</param>
    /// <param name="negativeDropIndex">Same as <paramref name="promptDropIndex"/> for the negative stream.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds,
        int[] negativeTokenIds,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null,
        int promptDropIndex = 0,
        int negativeDropIndex = 0)
    {
        ThrowIfDisposed();

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width;
        int height = request.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int hPacked = latentH / _config.PatchSize;
        int wPacked = latentW / _config.PatchSize;
        int imgSeqLen = hPacked * wPacked;
        int patchDim = _config.PatchSize * _config.PatchSize * _config.InChannels;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;
        bool useCfg = cfgScale > 1.0f;

        Logs.Info($"Qwen-Image: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        Backend.PreloadWeights(_textEncoder.EnumerateWeights());

        int[][] batchPrompt = [promptTokenIds];
        Tensor condHidden = _textEncoder.Encode(Backend, batchPrompt);
        if (promptDropIndex > 0)
        {
            Tensor trimmed = DropPrefixHiddenStates(condHidden, promptDropIndex);
            condHidden.Dispose();
            condHidden = trimmed;
        }

        Tensor? uncondHidden = null;
        if (useCfg)
        {
            int[][] batchNeg = [negativeTokenIds];
            uncondHidden = _textEncoder.Encode(Backend, batchNeg);
            if (negativeDropIndex > 0)
            {
                Tensor trimmed = DropPrefixHiddenStates(uncondHidden, negativeDropIndex);
                uncondHidden.Dispose();
                uncondHidden = trimmed;
            }
        }

        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms (txt seqLen={condHidden.Shape[1]})");

        Backend.FreeWeights(_textEncoder.EnumerateWeights());

        TensorShape latentShape = new TensorShape(1, _config.InChannels, latentH, latentW);
        TensorShape packedShape = new TensorShape(1, imgSeqLen, patchDim);

        FlowMatchEulerDiscreteScheduler scheduler =
            FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(imgSeqLen);
        scheduler.SetTimesteps(steps);

        Tensor unpackedNoise = SeedGenerator.CreateNoise(latentShape, seed);
        Tensor packedLatent = PackLatent(unpackedNoise, latentH, latentW, _config.InChannels, _config.PatchSize);
        unpackedNoise.Dispose();

        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(packedShape, DType.F32);
            Backend.Scale(scaled, packedLatent, initSigma);
            packedLatent.Dispose();
            packedLatent = scaled;
        }

        Backend.PreloadWeights(_transformer.EnumerateWeights());

        Logs.Info("Starting Qwen-Image denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];
            float normalizedT = t / 1000.0f;

            Tensor noisePred;
            if (useCfg)
            {
                Tensor condPred = _transformer.Forward(Backend, packedLatent, condHidden, normalizedT, hPacked, wPacked);
                Tensor uncondPred = _transformer.Forward(Backend, packedLatent, uncondHidden!, normalizedT, hPacked, wPacked);
                noisePred = CfgHelper.ApplyCfg(uncondPred, condPred, cfgScale);
                uncondPred.Dispose();
                condPred.Dispose();
            }
            else
            {
                noisePred = _transformer.Forward(Backend, packedLatent, condHidden, normalizedT, hPacked, wPacked);
            }

            Tensor newLatent = new Tensor(packedShape, DType.F32);
            scheduler.Step(newLatent, noisePred, packedLatent, i);
            noisePred.Dispose();
            packedLatent.Dispose();
            packedLatent = newLatent;

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        condHidden.Dispose();
        uncondHidden?.Dispose();

        QwenImageTransformer.DumpFinalLatent(packedLatent);

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        Tensor unpacked = UnpackLatent(packedLatent, latentH, latentW, _config.InChannels, _config.PatchSize);
        packedLatent.Dispose();

        VaeConfig vaeConfig = VaeConfig.QwenImage;
        Tensor scaled2 = new Tensor(unpacked.Shape, DType.F32);
        ApplyVaeShiftScale(scaled2, unpacked, vaeConfig.ShiftFactor ?? 0.0f, vaeConfig.ScalingFactor);
        unpacked.Dispose();

        Backend.PreloadWeights(_vaeDecoder.EnumerateWeights());
        Logs.Info("Decoding latents to image (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(Backend, scaled2);
        scaled2.Dispose();
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Qwen-Image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>Drops the first <paramref name="drop"/> sequence positions from a <c>[1, seq, hidden]</c>
    /// F32 hidden-state tensor, returning <c>[1, seq-drop, hidden]</c>. Used to discard the system+user-header
    /// template prefix from Qwen-Image text conditioning (the kept tail = prompt content + assistant suffix,
    /// matching diffusers' <c>split_hidden_states[drop_idx:]</c>). No-op clone guard if drop is out of range.</summary>
    private static Tensor DropPrefixHiddenStates(Tensor hidden, int drop)
    {
        long batch = hidden.Shape[0];
        long seq = hidden.Shape[1];
        long hiddenDim = hidden.Shape[2];
        long elem = hidden.DType.SizeInBytes;
        long rowBytes = hiddenDim * elem;
        // Guard: if the drop would empty the sequence (or batch isn't 1), copy the whole tensor unchanged
        // rather than drop — the caller always disposes the input and adopts our return value.
        long effectiveDrop = (drop <= 0 || drop >= seq || batch != 1) ? 0 : drop;
        long newSeq = seq - effectiveDrop;
        Tensor result = new Tensor(new TensorShape(batch, newSeq, hiddenDim), hidden.DType);
        byte* src = (byte*)hidden.DataPointer;
        byte* dst = (byte*)result.DataPointer;
        Buffer.MemoryCopy(src + effectiveDrop * rowBytes, dst, newSeq * rowBytes, newSeq * rowBytes);
        return result;
    }

    /// <summary>Packs a latent tensor from <c>[B, C, H, W]</c> to <c>[B, (H/p)*(W/p), p² * C]</c> by interleaving p×p spatial blocks into the feature dim. Diffusers uses <c>view → permute(0,2,4,1,3,5) → reshape</c>; we replicate the resulting layout where each p×p patch contributes channels C × p² to the token, ordered <c>(c, py, px)</c> so the inverse <see cref="UnpackLatent"/> is exact.</summary>
    private static Tensor PackLatent(Tensor latent, int h, int w, int channels, int patchSize)
    {
        int batch = (int)latent.Shape[0];
        int hPacked = h / patchSize;
        int wPacked = w / patchSize;
        int patchDim = channels * patchSize * patchSize;
        int seqLen = hPacked * wPacked;

        TensorShape packedShape = new TensorShape(batch, seqLen, patchDim);
        Tensor packed = new Tensor(packedShape, DType.F32);

        float* inPtr = (float*)latent.DataPointer;
        float* outPtr = (float*)packed.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int ph = 0; ph < hPacked; ph++)
            {
                for (int pw = 0; pw < wPacked; pw++)
                {
                    int seqIdx = ph * wPacked + pw;
                    long outBase = ((long)b * seqLen + seqIdx) * patchDim;

                    for (int c = 0; c < channels; c++)
                    {
                        long inChannelBase = ((long)b * channels + c) * h * w;
                        long patchBase = outBase + (long)c * patchSize * patchSize;

                        for (int py = 0; py < patchSize; py++)
                        {
                            for (int px = 0; px < patchSize; px++)
                            {
                                int row = ph * patchSize + py;
                                int col = pw * patchSize + px;
                                outPtr[patchBase + py * patchSize + px] = inPtr[inChannelBase + (long)row * w + col];
                            }
                        }
                    }
                }
            }
        }

        return packed;
    }

    /// <summary>Inverse of <see cref="PackLatent"/>.</summary>
    private static Tensor UnpackLatent(Tensor packed, int h, int w, int channels, int patchSize)
    {
        int batch = (int)packed.Shape[0];
        int hPacked = h / patchSize;
        int wPacked = w / patchSize;
        int patchDim = channels * patchSize * patchSize;
        int seqLen = hPacked * wPacked;

        TensorShape unpackedShape = new TensorShape(batch, channels, h, w);
        Tensor unpacked = new Tensor(unpackedShape, DType.F32);

        float* inPtr = (float*)packed.DataPointer;
        float* outPtr = (float*)unpacked.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int ph = 0; ph < hPacked; ph++)
            {
                for (int pw = 0; pw < wPacked; pw++)
                {
                    int seqIdx = ph * wPacked + pw;
                    long inBase = ((long)b * seqLen + seqIdx) * patchDim;

                    for (int c = 0; c < channels; c++)
                    {
                        long outChannelBase = ((long)b * channels + c) * h * w;
                        long patchBase = inBase + (long)c * patchSize * patchSize;

                        for (int py = 0; py < patchSize; py++)
                        {
                            for (int px = 0; px < patchSize; px++)
                            {
                                int row = ph * patchSize + py;
                                int col = pw * patchSize + px;
                                outPtr[outChannelBase + (long)row * w + col] = inPtr[patchBase + py * patchSize + px];
                            }
                        }
                    }
                }
            }
        }

        return unpacked;
    }

    /// <summary>Applies the diffusers-style VAE input transform: <c>(latent / scaling_factor) + shift_factor</c>. The decoder expects this de-normalized input; the scaling and shift constants come from the model card.</summary>
    private static void ApplyVaeShiftScale(Tensor output, Tensor input, float shift, float scale)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        long count = output.ElementCount;
        float invScale = 1.0f / scale;
        for (long i = 0; i < count; i++)
            outPtr[i] = inPtr[i] * invScale + shift;
    }
}
