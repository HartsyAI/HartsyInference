using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Ideogram 4 (<c>ideogram-oss/ideogram4</c>, 9.3B, non-commercial license) text-to-image pipeline. Orchestrates Qwen3-VL-8B (13-layer tap) → two single-stream DiTs (conditional + unconditional, asymmetric CFG) → Flux.2 VAE. Ported from upstream <c>pipeline_ideogram4.py</c>.
///
/// Pipeline-level specifics:
/// <list type="bullet">
///   <item><b>Unified sequence</b> per prompt: <c>[text tokens][image tokens]</c>. Text features (Qwen 13-layer concat) sit at text positions; noise sits at image positions; an <c>image_indicator</c> embedding + 3D MRoPE keep them apart.</item>
///   <item><b>Asymmetric CFG</b>: the positive pass runs the conditional transformer over the full sequence and keeps only the image-token velocity; the negative pass runs the <i>unconditional</i> transformer over an image-only sequence with zeroed text features. Combined as <c>v = gw·pos + (1−gw)·neg</c> with a per-step guidance schedule (gw≈7 main, gw≈3 polish).</item>
///   <item><b>Logit-normal schedule</b> (<see cref="LogitNormalSchedule"/>), resolution-adjusted mean, plain Euler <c>z += v·(s−t)</c>.</item>
///   <item><b>Fixed-constant latent norm</b> (<see cref="Ideogram4LatentNorm"/>) applied to the packed token latent before the 2×2 unpatchify — NOT the Flux.2 VAE BatchNorm.</item>
/// </list>
///
/// <para><b>VRAM:</b> both 9.3B transformers must be resident during the loop (each step runs both), so this needs roughly 2× the DiT footprint plus the VAE. The Qwen encoder is freed before the loop. Realistically a multi-GPU / high-VRAM host; documented in the Phase 4 checklist.</para></summary>
public sealed unsafe class Ideogram4Pipeline : DiffusionPipelineBase
{
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly Ideogram4Transformer _conditional;
    private readonly Ideogram4Transformer _unconditional;
    private readonly VaeDecoder _vaeDecoder;
    private readonly Ideogram4Config _config;

    private const int LlmTokenIndicator = 3;
    private const int OutputImageIndicator = 2;
    private const int ImagePositionOffset = 65536;

    /// <summary>Creates an Ideogram 4 pipeline from pre-loaded components.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="textEncoder">Qwen3-VL-8B language tower (<see cref="LlamaStyleEncoderConfig.Qwen3_VL_8B"/>).</param>
    /// <param name="conditional">Conditional transformer (<c>transformer/</c> weights).</param>
    /// <param name="unconditional">Unconditional transformer (<c>unconditional_transformer/</c> weights).</param>
    /// <param name="vaeDecoder">Flux.2 VAE decoder (<c>VaeConfig.Flux2</c>).</param>
    /// <param name="config">Architecture config (pinned to the loaded transformers).</param>
    public Ideogram4Pipeline(IBackend backend, LlamaStyleEncoder textEncoder,
        Ideogram4Transformer conditional, Ideogram4Transformer unconditional,
        VaeDecoder vaeDecoder, Ideogram4Config config)
        : base(backend)
    {
        _textEncoder = textEncoder;
        _conditional = conditional;
        _unconditional = unconditional;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image from chat-templated prompt token ids (the Qwen3 chat template must already be applied). The negative branch needs no tokens — Ideogram's CFG zeroes the text features.</summary>
    /// <param name="promptTokenIds">Tokenized, chat-templated prompt.</param>
    /// <param name="request">Width/Height/Seed (Steps and CfgScale come from <paramref name="preset"/>).</param>
    /// <param name="preset">Sampler preset (steps + guidance schedule + logit-normal mu/std). Defaults to <see cref="Ideogram4SamplerPreset.Default20"/>.</param>
    /// <param name="onProgress">Optional per-step callback.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds,
        TextToImageRequest request,
        Ideogram4SamplerPreset? preset = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        preset ??= Ideogram4SamplerPreset.Default20;

        if (promptTokenIds.Length == 0)
            throw new ArgumentException("Prompt token ids must be non-empty.", nameof(promptTokenIds));
        if (promptTokenIds.Length > _config.MaxTextTokens)
            throw new ArgumentException($"Prompt has {promptTokenIds.Length} tokens, exceeds MaxTextTokens={_config.MaxTextTokens}.", nameof(promptTokenIds));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int patch = _config.PatchSize * _config.VaeScaleFactor; // 2 × 8 = 16
        if (request.Width % patch != 0 || request.Height % patch != 0)
            throw new ArgumentException($"Width and height must be divisible by {patch} for Ideogram 4.");

        int gridH = request.Height / patch;
        int gridW = request.Width / patch;
        int numImageTokens = gridH * gridW;
        int numText = promptTokenIds.Length;
        int seqLen = numText + numImageTokens;
        int latentDim = _config.InChannels;
        int steps = preset.NumSteps;

        Logs.Info($"Ideogram 4: generating {request.Width}x{request.Height}, preset={preset.Name} ({steps} steps), seed={seed}");
        Logs.Warning("Ideogram 4 runs TWO 9.3B transformers concurrently (asymmetric CFG) — expect very high VRAM use.");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode prompt: Qwen3-VL 13-layer concat → [1, numText, llmFeaturesDim] ──
        Backend.PreloadWeights(_textEncoder.EnumerateWeights());
        Tensor textFeatures = _textEncoder.EncodeMultiLayer(Backend, [promptTokenIds], Ideogram4Config.QwenActivationLayersHf);
        if ((int)textFeatures.Shape[2] != _config.LlmFeaturesDim)
            throw new InvalidOperationException($"Encoder produced {textFeatures.Shape[2]}-dim features, expected {_config.LlmFeaturesDim} (13 × 4096). Check the tap-layer indices / encoder preset.");
        Backend.Sync();
        Backend.FreeWeights(_textEncoder.EnumerateWeights());
        Logs.Info($"Prompt encoded in {sw.ElapsedMilliseconds}ms");

        // ── 2. Build the unified-sequence conditioning tensors ──
        Tensor llmFull = PlaceTextFeatures(textFeatures, numText, seqLen, _config.LlmFeaturesDim);
        textFeatures.Dispose();
        Tensor posIds = BuildPositionIds(numText, gridH, gridW);              // [1, L, 3]
        int[] indicator = BuildIndicator(numText, numImageTokens);            // [L]
        Tensor posIdsImageOnly = SliceImagePositions(posIds, numText, numImageTokens);
        int[] indicatorImageOnly = new int[numImageTokens];
        Array.Fill(indicatorImageOnly, OutputImageIndicator);
        Tensor negLlm = new Tensor(new TensorShape(1, numImageTokens, _config.LlmFeaturesDim), DType.F32); // zeros

        // ── 3. Initial noise (image-token format) [1, nImg, 128] ──
        Tensor z = SeedGenerator.CreateNoise(new TensorShape(1, numImageTokens, latentDim), seed);

        // ── 4. Schedule ──
        LogitNormalSchedule schedule = LogitNormalSchedule.ForResolution(request.Height, request.Width, preset.Mu, preset.Std);
        float[] grid = LogitNormalSchedule.MakeStepIntervals(steps);

        // ── 5. Denoise loop (both transformers resident) ──
        Backend.PreloadWeights(_conditional.EnumerateWeights());
        Backend.PreloadWeights(_unconditional.EnumerateWeights());

        for (int i = steps - 1; i >= 0; i--)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float tVal = schedule.Map(grid[i + 1]);
            float sVal = schedule.Map(grid[i]);
            float delta = sVal - tVal;
            float gw = preset.GuidanceSchedule[i];

            // Positive pass: full sequence, keep image-token velocity.
            Tensor posX = BuildConditionalLatent(z, numText, numImageTokens, latentDim);
            Tensor posOut = _conditional.Forward(Backend, llmFull, posX, tVal, posIds, indicator, null);
            posX.Dispose();
            Tensor posV = SliceImageVelocity(posOut, numText, numImageTokens, latentDim);
            posOut.Dispose();

            // Negative pass: image-only sequence, zeroed text, unconditional transformer.
            Tensor negV = _unconditional.Forward(Backend, negLlm, z, tVal, posIdsImageOnly, indicatorImageOnly, null);

            // v = gw·pos + (1−gw)·neg ; z = z + v·delta
            CombineAndStep(z, posV, negV, gw, delta);
            posV.Dispose();
            negV.Dispose();

            stepSw.Stop();
            Logs.Debug($"Step {steps - i}/{steps} (t={tVal:F4}, gw={gw:F1}) in {stepSw.ElapsedMilliseconds}ms");
            // Tag the latent family for live-preview decoders (Flux.2 VAE, shared with Lens). The working
            // latent is token-packed [1, nImg, 128], so no per-step Latent snapshot is provided.
            onProgress?.Invoke(new GenerationProgress(steps - i, steps, stepSw.Elapsed.TotalMilliseconds)
            {
                LatentArch = LatentArchitecture.Flux2,
            });
        }

        llmFull.Dispose();
        posIds.Dispose();
        posIdsImageOnly.Dispose();
        negLlm.Dispose();

        // ── 6. Free DiT weights before VAE decode ──
        Backend.Sync();
        Backend.FreeWeights(_conditional.EnumerateWeights());
        Backend.FreeWeights(_unconditional.EnumerateWeights());

        // ── 7. Latent un-normalize (fixed constants) + unpatchify → [1, 32, H/8, W/8] ──
        ApplyLatentNorm(z);
        Tensor vaeIn = Unpatchify(z, gridH, gridW, _config.PatchSize);
        z.Dispose();

        // ── 8. VAE decode ──
        Logs.Verbose("Decoding latents (Flux.2 VAE, tiled)...");
        Tensor image = _vaeDecoder.DecodeTiled(Backend, vaeIn);
        vaeIn.Dispose();

        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Ideogram 4 generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (rgb, request.Width, request.Height, seed);
    }

    /// <summary>Places encoded text features <c>[1, numText, D]</c> into a full <c>[1, L, D]</c> tensor (text at the front, zeros at image positions).</summary>
    private static Tensor PlaceTextFeatures(Tensor textFeatures, int numText, int seqLen, int dim)
    {
        Tensor full = new Tensor(new TensorShape(1, seqLen, dim), DType.F32);
        float* dst = (float*)full.DataPointer;
        new Span<float>(dst, checked((int)((long)seqLen * dim))).Clear();
        float* src = (float*)textFeatures.DataPointer;
        long bytes = (long)numText * dim * sizeof(float);
        Buffer.MemoryCopy(src, dst, bytes, bytes);
        return full;
    }

    /// <summary>Builds <c>[1, L, 3]</c> MRoPE positions: text tokens <c>(i, i, i)</c>; image tokens <c>(0, row, col) + IMAGE_POSITION_OFFSET</c>.</summary>
    private static Tensor BuildPositionIds(int numText, int gridH, int gridW)
    {
        int numImg = gridH * gridW;
        int seqLen = numText + numImg;
        Tensor pos = new Tensor(new TensorShape(1, seqLen, 3), DType.F32);
        float* p = (float*)pos.DataPointer;
        for (int t = 0; t < numText; t++)
        {
            long off = (long)t * 3;
            p[off + 0] = t;
            p[off + 1] = t;
            p[off + 2] = t;
        }
        for (int idx = 0; idx < numImg; idx++)
        {
            int row = idx / gridW;
            int col = idx % gridW;
            long off = (long)(numText + idx) * 3;
            p[off + 0] = ImagePositionOffset;
            p[off + 1] = ImagePositionOffset + row;
            p[off + 2] = ImagePositionOffset + col;
        }
        return pos;
    }

    /// <summary>Slices the image-token rows <c>[numText..L)</c> out of a <c>[1, L, 3]</c> position tensor into <c>[1, numImg, 3]</c>.</summary>
    private static Tensor SliceImagePositions(Tensor posIds, int numText, int numImg)
    {
        Tensor outP = new Tensor(new TensorShape(1, numImg, 3), DType.F32);
        float* src = (float*)posIds.DataPointer;
        float* dst = (float*)outP.DataPointer;
        long bytes = (long)numImg * 3 * sizeof(float);
        Buffer.MemoryCopy(src + (long)numText * 3, dst, bytes, bytes);
        return outP;
    }

    private static int[] BuildIndicator(int numText, int numImg)
    {
        int[] ind = new int[numText + numImg];
        for (int t = 0; t < numText; t++) ind[t] = LlmTokenIndicator;
        for (int i = 0; i < numImg; i++) ind[numText + i] = OutputImageIndicator;
        return ind;
    }

    /// <summary>Builds the conditional-pass latent <c>[1, L, 128]</c> = <c>[zeros(numText) ; z]</c>.</summary>
    private static Tensor BuildConditionalLatent(Tensor z, int numText, int numImg, int dim)
    {
        int seqLen = numText + numImg;
        Tensor x = new Tensor(new TensorShape(1, seqLen, dim), DType.F32);
        float* dst = (float*)x.DataPointer;
        new Span<float>(dst, checked((int)((long)numText * dim))).Clear();
        float* src = (float*)z.DataPointer;
        long bytes = (long)numImg * dim * sizeof(float);
        Buffer.MemoryCopy(src, dst + (long)numText * dim, bytes, bytes);
        return x;
    }

    /// <summary>Slices image-token velocity rows <c>[numText..L)</c> out of <c>[1, L, 128]</c> into <c>[1, numImg, 128]</c>.</summary>
    private static Tensor SliceImageVelocity(Tensor full, int numText, int numImg, int dim)
    {
        Tensor outV = new Tensor(new TensorShape(1, numImg, dim), DType.F32);
        float* src = (float*)full.DataPointer;
        float* dst = (float*)outV.DataPointer;
        long bytes = (long)numImg * dim * sizeof(float);
        Buffer.MemoryCopy(src + (long)numText * dim, dst, bytes, bytes);
        return outV;
    }

    /// <summary>In-place Euler step on <paramref name="z"/>: <c>z += (gw·pos + (1−gw)·neg)·delta</c>.</summary>
    private static void CombineAndStep(Tensor z, Tensor posV, Tensor negV, float gw, float delta)
    {
        long n = z.Shape.ElementCount;
        float* zp = (float*)z.DataPointer;
        float* pp = (float*)posV.DataPointer;
        float* np = (float*)negV.DataPointer;
        float negW = 1.0f - gw;
        for (long i = 0; i < n; i++)
        {
            float v = gw * pp[i] + negW * np[i];
            zp[i] += v * delta;
        }
    }

    /// <summary>Applies the fixed per-channel latent norm in-place: <c>z[...,c] = z[...,c]·Scale[c] + Shift[c]</c> on a <c>[1, nImg, 128]</c> packed latent.</summary>
    private void ApplyLatentNorm(Tensor z)
    {
        int channels = (int)z.Shape[2];
        if (channels != Ideogram4LatentNorm.Channels)
            throw new InvalidOperationException($"Latent channels {channels} != {Ideogram4LatentNorm.Channels}.");
        long tokens = z.Shape[0] * z.Shape[1];
        float* zp = (float*)z.DataPointer;
        float[] scale = Ideogram4LatentNorm.Scale;
        float[] shift = Ideogram4LatentNorm.Shift;
        for (long tok = 0; tok < tokens; tok++)
        {
            long baseOff = tok * channels;
            for (int c = 0; c < channels; c++)
                zp[baseOff + c] = zp[baseOff + c] * scale[c] + shift[c];
        }
    }

    /// <summary>Unpatchifies the packed token latent <c>[1, nImg, patch²·aeC]</c> → <c>[1, aeC, gridH·patch, gridW·patch]</c> (upstream <c>view(B,gh,gw,p,p,ae).permute(0,5,1,3,2,4)</c>). For packed feature <c>f</c>: <c>ae = f % aeC, p2 = (f/aeC) % patch, p1 = f / (aeC·patch)</c>.</summary>
    private static Tensor Unpatchify(Tensor z, int gridH, int gridW, int patch)
    {
        int packedC = (int)z.Shape[2];
        int aeC = packedC / (patch * patch);
        int outH = gridH * patch;
        int outW = gridW * patch;
        Tensor outT = new Tensor(new TensorShape(1, aeC, outH, outW), DType.F32);
        float* src = (float*)z.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int gh = 0; gh < gridH; gh++)
        {
            for (int gw = 0; gw < gridW; gw++)
            {
                long token = (long)gh * gridW + gw;
                long srcBase = token * packedC;
                for (int p1 = 0; p1 < patch; p1++)
                {
                    for (int p2 = 0; p2 < patch; p2++)
                    {
                        int dstH = gh * patch + p1;
                        int dstW = gw * patch + p2;
                        for (int ae = 0; ae < aeC; ae++)
                        {
                            int f = p1 * (aeC * patch) + p2 * aeC + ae;
                            long dstOff = ((long)ae * outH + dstH) * outW + dstW;
                            dst[dstOff] = src[srcBase + f];
                        }
                    }
                }
            }
        }
        return outT;
    }
}
