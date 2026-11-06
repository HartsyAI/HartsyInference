using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Memory.Vram;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>SeedVR2 one-step video restoration. Per chunk: preprocess (bicubic area-resize IS the upscale —
/// the DiT only restores detail at target resolution) → VAE-encode → sample posterior → build the
/// 33-channel conditioning (noise‖latent‖mask=1) → ONE NaDiT forward at t=1000 (no CFG; cfg_scale 1.0 —
/// the negative embedding is loaded upstream but unused) → x₀ = noise − v → VAE-decode → optional wavelet
/// transplant → overlap cross-fade. Components are shared resources owned by the caller
/// (<see cref="DiffusionPipelineBase"/> convention).</summary>
public sealed class SeedVr2RestorePipeline : DiffusionPipelineBase
{
    private const float Timestep = 1000f;

    /// <summary>Working set per output pixel per frame for the untiled VAE encode and decode at F32 activations, calibrated on an
    /// RTX 2060 SUPER (8 GB): a 960×540 still completes and 1024×576 does not, with the DiT streamed.</summary>
    private const long BytesPerPixelFrameF32 = 9000;

    /// <summary>The same working set at BF16 activations, which the VAE runs where the backend has BF16: half the bytes.</summary>
    private const long BytesPerPixelFrameBf16 = 4500;

    private readonly SeedVr2Dit _dit;
    private readonly SeedVr2VaeEncoder _encoder;
    private readonly SeedVr2VaeDecoder _decoder;
    private readonly Tensor _posEmb;

    /// <summary>Parity hook: overrides RNG draws so E2E tests can inject the Python reference's saved
    /// noises (C# Box-Muller ≠ torch RNG — seeds can never match across runtimes). Kind is
    /// <c>"posterior"</c> or <c>"init"</c>; layout must be [cells,16] cell-major. Null = SeedGenerator.</summary>
    public Func<string, int, TensorShape, Tensor>? NoiseHook { get; set; }

    /// <summary>Wires shared components; <paramref name="posEmb"/> is the frozen positive embedding
    /// <c>[Lt,5120]</c> as F32 (cast from the shipped bf16 at load).</summary>
    public SeedVr2RestorePipeline(IBackend backend, SeedVr2Dit dit, SeedVr2VaeEncoder encoder,
        SeedVr2VaeDecoder decoder, Tensor posEmb)
        : base(backend)
    {
        _dit = dit;
        _encoder = encoder;
        _decoder = decoder;
        _posEmb = posEmb;
    }

    /// <summary>Restores RGB24 frames; returns restored RGB24 frames at the area-normalized target size.
    /// Stills (<c>frames.Count == 1</c>) flow through the same path with T=1.</summary>
    public (List<byte[]> Frames, int Width, int Height) Restore(IReadOnlyList<byte[]> frames, int width,
        int height, SeedVr2RestoreOptions options, Action<Core.Pipelines.GenerationProgress>? onProgress = null)
        => RestoreCore(frames, width, height, options, onProgress, null);

    /// <summary>Restores RGB24 frames while reporting each completed output chunk as directly displayable frames.</summary>
    public (List<byte[]> Frames, int Width, int Height) RestoreWithPreviews(IReadOnlyList<byte[]> frames, int width,
        int height, SeedVr2RestoreOptions options,
        Action<Core.Pipelines.GenerationProgress, IReadOnlyList<VideoFrame>> onProgress)
    {
        ArgumentNullException.ThrowIfNull(onProgress);
        return RestoreCore(frames, width, height, options, null, onProgress);
    }

    private (List<byte[]> Frames, int Width, int Height) RestoreCore(IReadOnlyList<byte[]> frames, int width,
        int height, SeedVr2RestoreOptions options, Action<Core.Pipelines.GenerationProgress>? onProgress,
        Action<Core.Pipelines.GenerationProgress, IReadOnlyList<VideoFrame>>? onPreview)
    {
        ThrowIfDisposed();
        long startTicks = Environment.TickCount64;

        long targetArea = FitTargetArea(options, frames.Count);
        SeedVr2Preprocess.Result pre = SeedVr2Preprocess.Run(frames, width, height, targetArea);
        int outH = pre.Height, outW = pre.Width;

        int chunkLen = Math.Min(SeedVr2Preprocess.PaddedFrameCount(Math.Max(options.ClipFrames, 1)), pre.Frames);
        int overlap = Math.Min(options.Overlap, Math.Max(chunkLen - 1, 0));
        int step = Math.Max(chunkLen - overlap, 1);

        float[] output = new float[3L * pre.Frames * outH * outW];
        float[] weight = new float[pre.Frames];
        int totalChunks = pre.Frames <= chunkLen ? 1 : (pre.Frames - chunkLen + step - 1) / step + 1;
        int chunkIndex = 0;

        for (int start = 0; start < pre.Frames; start += step)
        {
            int begin = Math.Min(start, Math.Max(pre.Frames - chunkLen, 0));
            int len = Math.Min(chunkLen, pre.Frames - begin);
            len = SeedVr2Preprocess.PaddedFrameCount(len) == len ? len : len - (len - 1) % 4;

            float[] chunk = RestoreChunk(pre, begin, len, options, chunkIndex);
            for (int f = 0; f < len; f++)
            {
                int dst = begin + f;
                // Linear cross-fade over overlapping frames (uniform accumulate + ramp weight).
                float ramp = weight[dst] > 0 && f < overlap ? (f + 1f) / (overlap + 1f) : 1f;
                for (int c = 0; c < 3; c++)
                {
                    long srcPlane = ((long)c * len + f) * outH * outW;
                    long dstPlane = ((long)c * pre.Frames + dst) * outH * outW;
                    if (weight[dst] == 0)
                        Array.Copy(chunk, srcPlane, output, dstPlane, (long)outH * outW);
                    else
                        for (long i = 0; i < (long)outH * outW; i++)
                            output[dstPlane + i] = output[dstPlane + i] * (1f - ramp) + chunk[srcPlane + i] * ramp;
                }
                weight[dst] = 1f;
            }
            chunkIndex++;
            Core.Pipelines.GenerationProgress update = new(chunkIndex, totalChunks,
                Environment.TickCount64 - startTicks);
            onProgress?.Invoke(update);
            if (onPreview is not null)
            {
                int previewCount = Math.Max(0, Math.Min(len, pre.OriginalFrames - begin));
                onPreview.Invoke(update, ConvertFrames(output, pre.Frames, begin, previewCount, outH, outW));
            }
            if (begin + len >= pre.Frames)
                break;
        }

        // One scratch for the whole restore: the colour passes reuse its planes for every frame.
        if (options.ColorTransfer != SeedVr2ColorTransfer.None || options.Strength < 1f)
        {
            using SeedVr2ColorScratch colorScratch = new(outH * outW);
            if (options.ColorTransfer != SeedVr2ColorTransfer.None)
            {
                ColorPass transfer = (model, input) => ApplyColorTransfer(options.ColorTransfer, model, input, outH, outW, colorScratch);
                for (int f = 0; f < pre.OriginalFrames; f++)
                    ColorFrame(output, pre.Data, pre.Frames, f, outH, outW, colorScratch, transfer);
            }
            if (options.Strength < 1f)
            {
                ColorPass wavelet = (model, input) => SeedVr2ColorFix.Transplant(model, input, 3, outH, outW, options.Strength, colorScratch);
                for (int f = 0; f < pre.OriginalFrames; f++)
                    ColorFrame(output, pre.Data, pre.Frames, f, outH, outW, colorScratch, wavelet);
            }
        }

        List<byte[]> result = ConvertFrames(output, pre.Frames, 0, pre.OriginalFrames, outH, outW)
            .Select(frame => frame.Rgb)
            .ToList();
        return (result, outW, outH);
    }

    private static List<VideoFrame> ConvertFrames(float[] source, int totalFrames, int start, int count,
        int height, int width)
    {
        List<VideoFrame> result = new List<VideoFrame>(count);
        for (int frame = start; frame < start + count; frame++)
        {
            byte[] rgb = new byte[height * width * 3];
            for (int c = 0; c < 3; c++)
            {
                long plane = ((long)c * totalFrames + frame) * height * width;
                for (int i = 0; i < height * width; i++)
                {
                    float v = (Math.Clamp(source[plane + i], -1f, 1f) * 0.5f + 0.5f) * 255f;
                    rgb[i * 3 + c] = (byte)Math.Clamp(MathF.Round(v), 0f, 255f);
                }
            }
            result.Add(new VideoFrame(frame, width, height, rgb));
        }
        return result;
    }

    /// <summary>Device bytes the DiT's forward needs beyond the weights it keeps resident: the streaming window (the block in
    /// flight plus the prefetched ones), the vid stream and MLP temporaries per token in F32, and a margin for the window
    /// attention's scratch. The window is reserved up front, since a resident prefix that takes all the free memory leaves the
    /// streamed suffix no room to upload (the first upload then fails on a 33 MB allocation on an 8 GB card).</summary>
    private long DitActivationReserveBytes(int lt, int lh, int lw)
    {
        SeedVr2Config config = _dit.Config;
        (int pT, int pH, int pW) = config.PatchSize;
        long tokens = (long)(lt / pT) * (lh / pH) * (lw / pW);
        long perToken = 4L * (12L * config.VidDim + 2L * config.MlpDim);
        long largestBlock = 0;
        for (int i = 0; i < _dit.BlockCount; i++)
            largestBlock = Math.Max(largestBlock, _dit.GetBlock(i).EstimatedWeightBytes);
        long window = (BlockStreamingOptions.DefaultPrefetchAhead + 1) * largestBlock;
        return tokens * perToken + window + VaeDecodeReserveBytes(lt, lh, lw) + 256L * 1024 * 1024;
    }

    /// <summary>The target area this restore can decode on this device: the requested area, or the largest area whose VAE decode
    /// fits the free memory after the DiT's streaming window and a fixed reserve.</summary>
    /// <remarks>The decode runs untiled, so its working set grows with the area. A large default target on a small card would
    /// otherwise fail partway through the restore. The fitted area is logged, and the returned frames carry the size used.</remarks>
    private long FitTargetArea(SeedVr2RestoreOptions options, int frameCount)
    {
        const long ReserveBytes = 1536L * 1024 * 1024;
        int chunk = Math.Min(SeedVr2Preprocess.PaddedFrameCount(Math.Max(options.ClipFrames, 1)), SeedVr2Preprocess.PaddedFrameCount(frameCount));
        long pixelFrames = chunk;
        long perPixel = pixelFrames * BytesPerPixelFrame;
        long fixedBytes = 384L * 1024 * 1024;

        Backend.TrimMemoryPool();
        long total = Backend.GetVramInfo().TotalBytes;
        if (total <= 0)
        {
            return options.TargetArea;
        }
        long free = Backend.FreeMemoryBytes();
        long usable = free - ReserveBytes - fixedBytes;
        long maxPixels = Math.Max(usable / perPixel, 256L * 256);
        long requested = options.TargetArea;
        if (requested <= maxPixels)
            return requested;

        Logs.Warning($"SeedVR2 restore target {requested} px needs ~{(requested * perPixel + fixedBytes) >> 20} MB to decode untiled, " +
            $"and {free >> 20} MB is free; restoring at {maxPixels} px instead.");
        return maxPixels;
    }

    /// <summary>The per-pixel figure for the activation dtype the VAE actually runs at.</summary>
    private long BytesPerPixelFrame =>
        _decoder.Config.ActivationDType == DType.BF16 ? BytesPerPixelFrameBf16 : BytesPerPixelFrameF32;

    /// <summary>Bytes the VAE decode of one chunk needs at the working precision. A still is one pixel frame; a clip's latent
    /// frame is four.</summary>
    private long VaeDecodeReserveBytes(int lt, int lh, int lw)
    {
        long pixelFrames = lt == 1 ? 1 : 4L * lt - 3;
        long pixels = (long)lh * 8 * lw * 8;
        return pixelFrames * pixels * BytesPerPixelFrame + 384L * 1024 * 1024;
    }

    private float[] RestoreChunk(SeedVr2Preprocess.Result pre, int begin, int len,
        SeedVr2RestoreOptions options, int chunkIndex)
    {
        int h = pre.Height, w = pre.Width;
        float scaling = _decoder.Config.ScalingFactor;

        Tensor pixels = new Tensor(new TensorShape([1L, 3, len, h, w]), DType.F32);
        Span<float> px = pixels.AsSpan<float>();
        for (int c = 0; c < 3; c++)
            pre.Data.AsSpan((int)(((long)c * pre.Frames + begin) * h * w), len * h * w)
                .CopyTo(px.Slice(c * len * h * w, len * h * w));

        // Phase-staged weight residency: only one component's weights stay on-device at a time — the VAE
        // activation peak wants the DiT's VRAM (and 12 GB cards need every phase alone). Measured: keeping
        // all three resident at 960×540-area peaked 23.98 GB for a ~0% wall win (uploads are not the
        // bottleneck; the DiT's rope/index/mod constants persist across chunks separately).
        long tEnc = Environment.TickCount64;
        Backend.PreloadWeights(_encoder.EnumerateWeights());
        (Tensor mean, Tensor logvar) = _encoder.Encode(Backend, pixels);
        Backend.Sync();
        Backend.FreeWeights(_encoder.EnumerateWeights());
        Backend.TrimMemoryPool();
        pixels.Dispose();
        tEnc = Environment.TickCount64 - tEnc;

        int lt = (int)mean.Shape[2], lh = (int)mean.Shape[3], lw = (int)mean.Shape[4];
        int cells = lt * lh * lw;
        const int C = 16;

        // Posterior sample (use_sample: True) then ×scaling, laid out channels-last [t',h',w',16].
        Tensor noiseA = NoiseHook?.Invoke("posterior", chunkIndex, new TensorShape(cells, C))
            ?? SeedGenerator.CreateNoise(new TensorShape(cells, C), options.Seed * 977 + chunkIndex * 131);
        Tensor noiseB = NoiseHook?.Invoke("init", chunkIndex, new TensorShape(cells, C))
            ?? SeedGenerator.CreateNoise(new TensorShape(cells, C), options.Seed * 977 + chunkIndex * 131 + 7);
        ReadOnlySpan<float> m = mean.AsSpan<float>(), lv = logvar.AsSpan<float>();
        ReadOnlySpan<float> epsA = noiseA.AsSpan<float>(), initNoise = noiseB.AsSpan<float>();

        Tensor ditIn = new Tensor(new TensorShape([(long)lt, lh, lw, 33]), DType.F32);
        Span<float> di = ditIn.AsSpan<float>();
        for (int cell = 0; cell < cells; cell++)
        {
            int t0 = cell / (lh * lw), rem = cell % (lh * lw);
            for (int c = 0; c < C; c++)
            {
                int chw = (c * lt + t0) * lh * lw + rem;
                float std = MathF.Exp(0.5f * Math.Clamp(lv[chw], -30f, 20f));
                float z = (m[chw] + std * epsA[cell * C + c]) * scaling;
                di[cell * 33 + c] = initNoise[cell * C + c];
                di[cell * 33 + C + c] = z;
            }
            di[cell * 33 + 32] = 1f;
        }
        mean.Dispose(); logvar.Dispose(); noiseA.Dispose();

        long tDit = Environment.TickCount64;
        // The DiT runs through the shared streaming scope: when the VRAM policy says it does not fit, each layer is made
        // resident just before it runs and evicted after, so a card too small for the whole model still restores. When it
        // fits, the scope preloads everything, which is the old full-residency path.
        Tensor v;
        using (BlockStreamingScope stream = BlockStreamingScope.Open(new BlockStreamingOptions
        {
            Backend = Backend,
            Denoiser = _dit,
            ModelName = "SeedVR2",
            HeadroomBytes = DitActivationReserveBytes(lt, lh, lw),
        }))
        {
            v = _dit.Forward(Backend, ditIn, _posEmb, Timestep);
            Backend.Sync();
            stream.EndStep();
        }
        Backend.FreeWeights(_dit.EnumerateWeights());
        Backend.TrimMemoryPool();
        ditIn.Dispose();
        tDit = Environment.TickCount64 - tDit;

        // One Euler step from σ=1: x₀ = x_t − v with x_t = the init noise; then ÷scaling, channels-first.
        Tensor latent = new Tensor(new TensorShape([1L, C, lt, lh, lw]), DType.F32);
        Span<float> ls = latent.AsSpan<float>();
        ReadOnlySpan<float> vs = v.AsSpan<float>();
        for (int cell = 0; cell < cells; cell++)
        {
            int t0 = cell / (lh * lw), rem = cell % (lh * lw);
            for (int c = 0; c < C; c++)
                ls[(c * lt + t0) * lh * lw + rem] = (initNoise[cell * C + c] - vs[cell * C + c]) / scaling;
        }
        v.Dispose(); noiseB.Dispose();

        long tDec = Environment.TickCount64;
        Backend.PreloadWeights(_decoder.EnumerateWeights());
        Tensor decoded = _decoder.Decode(Backend, latent);
        Backend.Sync();
        Backend.FreeWeights(_decoder.EnumerateWeights());
        Backend.TrimMemoryPool();
        latent.Dispose();
        float[] chunk = decoded.AsSpan<float>().ToArray();
        decoded.Dispose();
        tDec = Environment.TickCount64 - tDec;
        Logs.Debug($"SeedVR2 chunk {chunkIndex} ({len}f): encode {tEnc} ms, dit {tDit} ms, decode {tDec} ms");
        return chunk;
    }

    /// <summary>One colour pass over a frame's planes: <c>model</c> is overwritten, <c>input</c> is the resized reference.</summary>
    private delegate void ColorPass(Span<float> model, ReadOnlySpan<float> input);

    private static void ApplyColorTransfer(SeedVr2ColorTransfer mode, Span<float> model, ReadOnlySpan<float> input, int h, int w,
        SeedVr2ColorScratch scratch)
    {
        switch (mode)
        {
            case SeedVr2ColorTransfer.Lab:
                SeedVr2ColorFix.TransferLab(model, input, h, w, scratch);
                break;
            case SeedVr2ColorTransfer.AdaIn:
                SeedVr2ColorFix.TransferAdaIn(model, input, 3, h, w);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(mode), mode, "No colour transfer to apply.");
        }
    }

    /// <summary>Runs one colour pass over a frame: copies its three planes out of the clip-major output into the scratch, applies
    /// <paramref name="transfer"/>, and writes them back.</summary>
    private static void ColorFrame(float[] output, float[] reference, int totalFrames, int frame, int h, int w,
        SeedVr2ColorScratch scratch, ColorPass transfer)
    {
        int n = h * w;
        Span<float> model = scratch.Get(SeedVr2ColorScratch.Region.FrameModel);
        Span<float> input = scratch.Get(SeedVr2ColorScratch.Region.FrameInput);
        for (int c = 0; c < 3; c++)
        {
            long plane = ((long)c * totalFrames + frame) * n;
            output.AsSpan((int)plane, n).CopyTo(model.Slice(c * n, n));
            reference.AsSpan((int)plane, n).CopyTo(input.Slice(c * n, n));
        }
        transfer(model, input);
        for (int c = 0; c < 3; c++)
            model.Slice(c * n, n).CopyTo(output.AsSpan((int)(((long)c * totalFrames + frame) * n), n));
    }
}
