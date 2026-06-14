using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;
using SharpInference.Video;

namespace SharpInference.Interactive.Pipelines;

/// <summary>Matrix-Game 2.0 (Skywork, MIT) image-to-controllable-video pipeline — the few-step autoregressive loop of
/// <c>pipeline/causal_inference.py</c>: the seed image is VAE-encoded into a channel-concat condition (16ch latent +
/// 4ch first-frame mask → in_dim 36 with the noisy latent) and CLIP-ViT-H visual context (cross-attention K/V), then
/// blocks of 3 latent frames are generated with the DMD-distilled 3/4-step <see cref="FlowMatchDmdScheduler"/>,
/// each forward seeing the sliding window of previously generated clean frames at t=0 (Self-Forcing context).
/// Construct with <see cref="MatrixGame2Config.Universal"/> / <see cref="MatrixGame2Config.GtaDrive"/> /
/// <see cref="MatrixGame2Config.TempleRun"/> for the per-domain variants.
///
/// <para><b>Status: built, first-run validation pending.</b> Documented approximations until validation: context
/// frames are recomputed per forward instead of KV-cached (equivalent up to window-edge effects), and the
/// image-condition latent for padding frames encodes one black frame instead of the chunked causal video encode.</para></summary>
public sealed unsafe class MatrixGame2Pipeline : DiffusionPipelineBase
{
    private static readonly float[] _clipMean = [0.48145466f, 0.4578275f, 0.40821073f];
    private static readonly float[] _clipStd = [0.26862954f, 0.26130258f, 0.27577711f];

    private readonly MatrixGame2Transformer _transformer;
    private readonly Wan21VaeDecoder _vae;
    private readonly Wan21VaeEncoder _encoder;
    private readonly ClipVisionEncoder? _clip;
    private readonly MatrixGame2Config _config;

    public MatrixGame2Pipeline(IBackend backend, MatrixGame2Transformer transformer, Wan21VaeDecoder vae,
        Wan21VaeEncoder encoder, MatrixGame2Config config, ClipVisionEncoder? clip = null)
        : base(backend)
    {
        _transformer = transformer;
        _vae = vae;
        _encoder = encoder;
        _config = config;
        _clip = clip;
    }

    /// <summary>RGB-rate action rows required for <paramref name="totalLatentFrames"/>.</summary>
    public int RequiredActionFrames(int totalLatentFrames) => (totalLatentFrames - 1) * _config.VaeTemporalCompression + 1;

    /// <summary>Encodes the seed image into the CLIP visual context <c>[257, 1280]</c> (bilinear resize to 224² +
    /// CLIP normalization → penultimate hidden states). Requires the pipeline to be constructed with a
    /// <see cref="ClipVisionEncoder"/> (ViT-H/14). The caller owns the returned tensor.</summary>
    public Tensor EncodeClipContext(ReadOnlySpan<byte> rgb24, int width, int height)
    {
        ThrowIfDisposed();
        if (_clip is null)
            throw new InvalidOperationException("CLIP context encoding needs a ClipVisionEncoder (ViT-H/14) — construct the pipeline with one, or pass a precomputed context.");
        int size = _clip.Config.ImageSize;
        Tensor pixels = new Tensor(new TensorShape([1L, 3, size, size]), DType.F32);
        float* pp = (float*)pixels.DataPointer;
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    float sx = (x + 0.5f) * width / size - 0.5f;
                    float sy = (y + 0.5f) * height / size - 0.5f;
                    int x0 = Math.Clamp((int)MathF.Floor(sx), 0, width - 1), x1 = Math.Min(x0 + 1, width - 1);
                    int y0 = Math.Clamp((int)MathF.Floor(sy), 0, height - 1), y1 = Math.Min(y0 + 1, height - 1);
                    float fx = Math.Clamp(sx - x0, 0f, 1f), fy = Math.Clamp(sy - y0, 0f, 1f);
                    float v00 = rgb24[(y0 * width + x0) * 3 + c], v01 = rgb24[(y0 * width + x1) * 3 + c];
                    float v10 = rgb24[(y1 * width + x0) * 3 + c], v11 = rgb24[(y1 * width + x1) * 3 + c];
                    float v = (v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy;
                    pp[((long)c * size + y) * size + x] = (v / 255f - _clipMean[c]) / _clipStd[c];
                }
        Tensor hidden = _clip.EncodeHiddenStates(Backend, pixels);
        pixels.Dispose();
        Tensor context = CfgHelper.SliceBatchElement(hidden, 0, (int)hidden.Shape[1], (int)hidden.Shape[2]);
        hidden.Dispose();
        return context;
    }

    /// <summary>Generates a controllable rollout from an RGB24 seed image and per-RGB-frame action rows
    /// (<paramref name="keyboard"/> one-hot [K], <paramref name="mouse"/> (pitchΔ, yawΔ) — null for TempleRun).
    /// <paramref name="clipContext"/> is the CLIP visual context (<see cref="EncodeClipContext"/> or precomputed).
    /// Returns one interleaved-RGB <c>byte[]</c> per frame ((T−1)·4 + 1 frames).</summary>
    public (byte[][] frames, int width, int height, int seed) Generate(
        ReadOnlySpan<byte> seedRgb24, int width, int height, Tensor clipContext,
        float[][] keyboard, float[][]? mouse, int totalLatentFrames, int? seed = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        int sp = _config.VaeSpatialCompression * _config.PatchSize.H;   // 16: VAE 8× · patch 2
        if (width % sp != 0 || height % sp != 0)
            throw new ArgumentException($"Width/height must be divisible by {sp} for Matrix-Game 2.");
        int blockSize = _config.NumFramePerBlock;
        if (totalLatentFrames < blockSize || totalLatentFrames % blockSize != 0)
            throw new ArgumentException($"totalLatentFrames must be a positive multiple of {blockSize}.", nameof(totalLatentFrames));
        int requiredActions = RequiredActionFrames(totalLatentFrames);
        if (keyboard.Length < requiredActions)
            throw new ArgumentException($"Need ≥{requiredActions} keyboard rows; got {keyboard.Length}.", nameof(keyboard));
        if (_config.EnableMouse && (mouse is null || mouse.Length < requiredActions))
            throw new ArgumentException($"Need ≥{requiredActions} mouse rows for this variant.", nameof(mouse));

        int actualSeed = seed ?? SeedGenerator.RandomSeed();
        int hLat = height / _config.VaeSpatialCompression, wLat = width / _config.VaeSpatialCompression;
        int numBlocks = totalLatentFrames / blockSize;

        Logs.Info($"Matrix-Game 2: {totalLatentFrames} latent frames ({numBlocks} blocks) {width}x{height}, " +
            $"{_config.DenoisingStepList.Length} steps/block, kbd {_config.KeyboardDim}-dim, mouse={_config.EnableMouse}, seed={actualSeed}");
        Logs.Warning("Matrix-Game 2 pipeline is first-run-validation pending — numerics unverified vs the reference.");

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        FlowMatchDmdScheduler scheduler = new(_config.DenoisingStepList, _config.TimestepShift);

        // Image-condition latents: seed frame 0, one encoded black frame replicated for the padding positions.
        Tensor seedLatent = _encoder.EncodeRgbFrame(Backend, seedRgb24, width, height);
        Tensor blackLatent = EncodeBlackFrame(width, height);

        List<Tensor> generated = new(totalLatentFrames);   // clean [1,16,1,h,w] frames
        try
        {
            for (int b = 0; b < numBlocks; b++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                int blockStart = b * blockSize;
                int ctxCount = Math.Min(generated.Count, _config.LocalAttnSize <= 0 ? generated.Count : _config.LocalAttnSize);
                int firstFrame = blockStart - ctxCount;
                int included = ctxCount + blockSize;

                int[] ropeIdx = new int[included];
                for (int i = 0; i < included; i++) ropeIdx[i] = Math.Min(firstFrame + i, _config.RopeMaxSeqLen - 1);
                float[] frameTs = new float[included];

                Tensor current = SeedGenerator.CreateNoise(new TensorShape([1L, 16, blockSize, hLat, wLat]), actualSeed + b);
                Tensor mouseRows = BuildActionRows(mouse, firstFrame, included, 2, requiredActions);
                Tensor kbdRows = BuildActionRows(keyboard, firstFrame, included, _config.KeyboardDim, requiredActions);

                for (int k = 0; k < scheduler.NumSteps; k++)
                {
                    for (int i = 0; i < included; i++) frameTs[i] = i < ctxCount ? 0f : scheduler.Timestep(k);
                    Tensor input = AssembleInput(generated, current, seedLatent, blackLatent, firstFrame, ctxCount, blockSize, hLat, wLat);
                    Tensor v = _transformer.Forward(Backend, input, clipContext, frameTs, ropeIdx, blockSize, mouseRows, kbdRows);
                    input.Dispose();

                    scheduler.ToX0(current, v, k);
                    v.Dispose();
                    if (k < scheduler.NumSteps - 1)
                    {
                        Tensor fresh = SeedGenerator.CreateNoise(current.Shape, actualSeed + b * 100 + k + 1);
                        scheduler.AddNoise(current, fresh, k + 1);
                        fresh.Dispose();
                    }
                }

                for (int f = 0; f < blockSize; f++) generated.Add(SliceFrame(current, f, hLat, wLat));
                current.Dispose();
                mouseRows.Dispose();
                kbdRows.Dispose();
                sw.Stop();
                onProgress?.Invoke(new GenerationProgress(b + 1, numBlocks, sw.Elapsed.TotalMilliseconds));
            }

            Backend.Sync();
            Backend.FreeWeights(_transformer.EnumerateWeights());

            Tensor allLatents = ConcatFrames(generated, hLat, wLat);
            Tensor rgb = _vae.Decode(Backend, allLatents);
            allLatents.Dispose();
            int frames = (int)rgb.Shape[2];
            byte[][] output = new byte[frames][];
            for (int i = 0; i < frames; i++) output[i] = VideoRgbFrames.ExtractFrame(rgb, i);
            rgb.Dispose();

            Logs.Info($"Matrix-Game 2 rollout complete ({output.Length} frames, seed={actualSeed})");
            return (output, width, height, actualSeed);
        }
        finally
        {
            foreach (Tensor t in generated) t.Dispose();
            seedLatent.Dispose();
            blackLatent.Dispose();
        }
    }

    /// <summary>Channel-concat model input <c>[1, 36, included, h, w]</c>: [0..15] = clean context ‖ current noisy;
    /// [16..19] = first-frame mask; [20..35] = image-condition latent (seed at global frame 0, black elsewhere).</summary>
    private Tensor AssembleInput(List<Tensor> generated, Tensor current, Tensor seedLatent, Tensor blackLatent,
        int firstFrame, int ctxCount, int blockSize, int h, int w)
    {
        int included = ctxCount + blockSize;
        Tensor input = new Tensor(new TensorShape([1L, 36, included, h, w]), DType.F32);
        float* ip = (float*)input.DataPointer;
        long frame = (long)h * w;
        new Span<float>(ip, (int)Math.Min(int.MaxValue, input.Shape.ElementCount)).Clear();

        for (int i = 0; i < included; i++)
        {
            int globalFrame = firstFrame + i;
            // Latent channels 0..15.
            float* src = i < ctxCount
                ? (float*)generated[globalFrame].DataPointer
                : (float*)current.DataPointer + (long)(i - ctxCount) * frame;   // per-channel offset added below
            for (int c = 0; c < 16; c++)
            {
                long dst = ((long)c * included + i) * frame;
                float* channelSrc = i < ctxCount
                    ? (float*)generated[globalFrame].DataPointer + (long)c * frame
                    : (float*)current.DataPointer + ((long)c * blockSize + (i - ctxCount)) * frame;
                Buffer.MemoryCopy(channelSrc, ip + dst, frame * 4, frame * 4);
            }
            _ = src;
            // Mask channels 16..19: 1 on global frame 0.
            if (globalFrame == 0)
                for (int c = 16; c < 20; c++)
                    new Span<float>(ip + ((long)c * included + i) * frame, (int)frame).Fill(1f);
            // Image-condition channels 20..35.
            float* condSrc = (float*)(globalFrame == 0 ? seedLatent : blackLatent).DataPointer;
            for (int c = 0; c < 16; c++)
                Buffer.MemoryCopy(condSrc + (long)c * frame, ip + ((long)(20 + c) * included + i) * frame, frame * 4, frame * 4);
        }
        return input;
    }

    private Tensor EncodeBlackFrame(int width, int height)
    {
        byte[] black = new byte[width * height * 3];
        return _encoder.EncodeRgbFrame(Backend, black, width, height);
    }

    /// <summary>RGB-rate action rows covering the included latent frames (<c>included·4</c> rows from
    /// <c>firstFrame·4</c>, clamped into the plan).</summary>
    private Tensor BuildActionRows(float[][]? rows, int firstFrame, int includedFrames, int dim, int planLength)
    {
        int count = includedFrames * _config.VaeTemporalCompression;
        Tensor o = new Tensor(new TensorShape(count, dim), DType.F32);
        float* p = (float*)o.DataPointer;
        if (rows is null) return o;   // zeros (mouse-disabled variants)
        int start = firstFrame * _config.VaeTemporalCompression;
        for (int i = 0; i < count; i++)
        {
            float[] row = rows[Math.Clamp(start + i, 0, Math.Min(planLength, rows.Length) - 1)];
            for (int d = 0; d < dim && d < row.Length; d++) p[(long)i * dim + d] = row[d];
        }
        return o;
    }

    private static Tensor SliceFrame(Tensor block, int frameIndex, int h, int w)
    {
        int c = (int)block.Shape[1], t = (int)block.Shape[2];
        Tensor o = new Tensor(new TensorShape([1L, c, 1, h, w]), DType.F32);
        float* src = (float*)block.DataPointer;
        float* dst = (float*)o.DataPointer;
        long frame = (long)h * w;
        for (int ci = 0; ci < c; ci++)
            Buffer.MemoryCopy(src + ((long)ci * t + frameIndex) * frame, dst + (long)ci * frame, frame * 4, frame * 4);
        return o;
    }

    private static Tensor ConcatFrames(List<Tensor> frames, int h, int w)
    {
        int c = (int)frames[0].Shape[1], t = frames.Count;
        Tensor o = new Tensor(new TensorShape([1L, c, t, h, w]), DType.F32);
        float* dst = (float*)o.DataPointer;
        long frame = (long)h * w;
        for (int ci = 0; ci < c; ci++)
            for (int f = 0; f < t; f++)
                Buffer.MemoryCopy((float*)frames[f].DataPointer + (long)ci * frame, dst + ((long)ci * t + f) * frame, frame * 4, frame * 4);
        return o;
    }
}
