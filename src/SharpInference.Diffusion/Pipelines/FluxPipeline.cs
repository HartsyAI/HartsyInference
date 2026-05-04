using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.MemoryManagement;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>Flux text-to-image and image-to-image pipeline. Orchestrates CLIP-L pooled + T5-XXL text encoding → FluxTransformer denoising with flow matching → VAE decode → RGB image output. Supports Dev (guidance embedding) and Schnell (distilled, 1-4 steps) modes.
/// <para>Img2img is selected by passing an <see cref="ImageToImageRequest"/> (instead of <see cref="TextToImageRequest"/>) to <see cref="GenerateFromTokens"/>. Requires a <see cref="VaeEncoder"/> on construction. The img2img path encodes the source via the 16-channel Flux VAE, packs the latent (2×2 patchify), and injects flow-matching noise at the timestep selected by <c>Strength</c>.</para>
/// </summary>
public sealed unsafe class FluxPipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly ClipTextEncoder _clipL;
    private readonly T5TextEncoder _t5;
    private readonly FluxTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly FluxConfig _config;
    private int _disposed;

    /// <summary>Creates a new Flux pipeline. Img2img is unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    public FluxPipeline(IBackend backend, ClipTextEncoder clipL, T5TextEncoder t5,
        FluxTransformer transformer, VaeDecoder vaeDecoder, FluxConfig config)
        : this(backend, clipL, t5, transformer, vaeDecoder, vaeEncoder: null, config)
    {
    }

    /// <summary>Creates a new Flux pipeline with both VAE halves loaded. Required for img2img and for use as a cross-model refiner.</summary>
    public FluxPipeline(IBackend backend, ClipTextEncoder clipL, T5TextEncoder t5,
        FluxTransformer transformer, VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder, FluxConfig config)
    {
        _backend = backend;
        _clipL = clipL;
        _t5 = t5;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-tokenized input. Handles both text-to-image and image-to-image via the runtime type of <paramref name="request"/>:
    /// <list type="bullet">
    /// <item>Plain <see cref="TextToImageRequest"/> → text-to-image (initial packed latent = noise scaled by initSigma; denoise from step 0).</item>
    /// <item><see cref="ImageToImageRequest"/> → image-to-image. The source image is encoded via the 16-channel Flux VAE, packed (2×2 patchify), and combined with fresh packed noise via flow-matching <c>AddNoise</c> at <c>sigma[startStep]</c>. Requires a <see cref="VaeEncoder"/>.</item>
    /// </list>
    /// Strength=0 short-circuits to byte-identical pass-through.
    /// </summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsL,
        int promptEosPositionL,
        int[] promptTokenIdsT5,
        int[]? promptAttentionMaskT5,
        TextToImageRequest request,
        float guidanceScale = 3.5f,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width;
        int height = request.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int steps = request.Steps;

        // Img2img validation + strength=0 short-circuit BEFORE any model work.
        int startStep = 0;
        if (request is ImageToImageRequest img2img)
        {
            Tensor src = img2img.SourceImage;
            if (src.Shape.Rank != 4 || src.Shape[0] != 1 || src.Shape[1] != 3 ||
                src.Shape[2] != height || src.Shape[3] != width)
            {
                throw new ArgumentException(
                    $"SourceImage shape must be [1, 3, {height}, {width}] (matching request); got {src.Shape}.",
                    nameof(request));
            }

            float strength = Math.Clamp(img2img.Strength, 0f, 1f);
            int initTimesteps = (int)MathF.Round(steps * strength);
            startStep = Math.Max(steps - initTimesteps, 0);

            if (initTimesteps == 0)
            {
                Logs.Info("Strength=0; passing source through unchanged");
                return (ImagePostProcessor.TensorToRgbBytes(src), width, height, seed);
            }
        }

        string baseMode = _config.GuidanceEmbed ? "Dev" : "Schnell";
        string opMode = isImg2Img ? $"img2img (start={startStep}/{steps})" : "txt2img";
        Logs.Info($"Flux ({baseMode}) {opMode}: {width}x{height}, {steps} steps, guidance={guidanceScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode text ─────────────────────────────────────────────
        Logs.Info("Encoding text with CLIP-L (pooled) + T5-XXL (per-token)...");

        // Preload T5 weights to GPU as a single batch upload. Without this, every
        // matmul/layernorm inside the encoder would do its own cache-miss H2D
        // transfer + immediate free (see CudaBackend.MatMul finally block) — turning
        // text encoding into thousands of ~MB-sized PCIe ping-pongs instead of one
        // bulk transfer + many on-GPU reuses. Backends that don't support a weight
        // cache (Cpu, Vulkan) treat PreloadWeights as a no-op.
        _backend.PreloadWeights(_t5.EnumerateWeights());

        int[][] batchTokenIdsL = [promptTokenIdsL];
        Tensor clipLHidden = _clipL.Encode(_backend, batchTokenIdsL);
        LogTensorStats("CLIP hidden (full)", clipLHidden);
        Tensor clipPooled = ExtractEosHiddenState(clipLHidden, promptEosPositionL);
        clipLHidden.Dispose();

        int[][] batchTokenIdsT5 = [promptTokenIdsT5];
        int[][]? batchMaskT5 = promptAttentionMaskT5 is not null ? [promptAttentionMaskT5] : null;
        Tensor t5Embeddings = _t5.Encode(_backend, batchTokenIdsT5, batchMaskT5);

        int txtSeqLen = (int)t5Embeddings.Shape[1];
        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms (T5 seqLen={txtSeqLen})");
        LogTensorStats("CLIP pooled", clipPooled);
        LogTensorStats("T5 embeddings", t5Embeddings);

        // Free T5 weights from GPU now that text encoding is done. T5-XXL is ~5 GB —
        // keeping it cached through sampling + VAE decode would OOM 12 GB cards on Flux.
        // The activation tensor `t5Embeddings` is on GPU/CPU and lives independently of
        // the encoder weights. Re-uploading T5 on the next generation is a one-time cost
        // (~hundreds of ms) that's strictly cheaper than swapping mid-pipeline.
        _backend.FreeWeights(_t5.EnumerateWeights());

        // ── 2. Set up dynamic flow-match scheduler ─────────────────────
        TensorShape latentShape = new TensorShape(1, 16, latentH, latentW);
        int hPacked = latentH / 2;
        int wPacked = latentW / 2;
        int imgSeqLen = hPacked * wPacked;
        TensorShape packedShape = new TensorShape(1, imgSeqLen, 64);

        FlowMatchEulerDiscreteScheduler scheduler =
            FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(imgSeqLen);
        scheduler.SetTimesteps(steps);

        // ── 3. Build initial packed latent ─────────────────────────────
        // T2I: noise scaled by initSigma. Img2img: vaeEncoder.Encode → Pack → AddNoise at sigma[startStep].
        Tensor packedLatent = BuildInitialPackedLatent(request, scheduler, latentShape, packedShape, latentH, latentW, seed, startStep);

        // ── 4. Denoising loop ─────────────────────────────────────────
        // Two paths depending on whether the backend can stream:
        //   - StreamingCache != null (CUDA): use BlockStreamingController so resident
        //     VRAM peaks at ~(activations + 2 blocks of weights), making Flux work on
        //     12 GB cards. Shared (non-block) transformer weights still preload eagerly
        //     since they're touched on every step and only ~80 MB total.
        //   - StreamingCache == null (CPU/Vulkan): preload everything eagerly. CPU has
        //     no notion of "device memory"; Vulkan's allocator is independent of this API.
        //     Same behavior as before this refactor.
        BlockStreamingController? streamer = null;
        if (_backend.StreamingCache is not null)
        {
            _backend.PreloadWeights(_transformer.EnumerateSharedWeights());
            IStreamingBlock[] blocks = new IStreamingBlock[_transformer.BlockCount];
            for (int b = 0; b < blocks.Length; b++) blocks[b] = _transformer.GetBlock(b);

            int prefetchAhead = ChooseFluxPrefetchAhead(
                _backend.StreamingCache, blocks, txtSeqLen, imgSeqLen, _config.HiddenSize, (int)(_config.HiddenSize * _config.MlpRatio));
            streamer = new BlockStreamingController(_backend.StreamingCache, blocks, prefetchAhead: prefetchAhead, retainBehind: 0);
            _transformer.BeforeBlockForward = streamer.BeforeBlockForward;
            streamer.Prime();
            long perBlockMb = streamer.EstimatedTotalWeightBytes / blocks.Length / (1024 * 1024);
            long totalMb = streamer.EstimatedTotalWeightBytes / (1024 * 1024);
            Logs.Info($"Flux streaming: {blocks.Length} blocks, prefetchAhead={prefetchAhead}, " +
                $"per-block ~{perBlockMb} MB, total ~{totalMb} MB");
        }
        else
        {
            _backend.PreloadWeights(_transformer.EnumerateWeights());
        }

        Logs.Info("Starting Flux denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = timesteps[i] / 1000.0f; // Convert timestep back to sigma [0,1]

            // Forward pass: velocity prediction
            Tensor velocityPred = _transformer.Forward(
                _backend, packedLatent, t5Embeddings, sigma,
                clipPooled!, guidanceScale, txtSeqLen, hPacked, wPacked);

            LogTensorStats($"Step {i+1} velocity", velocityPred);
            LogPerLatentChannelMeanPacked($"Step {i+1} velocity", velocityPred);

            // Scheduler step: Euler on packed latent
            TensorShape packedStepShape = new TensorShape(1, imgSeqLen, 64);
            Tensor newLatent = new Tensor(packedStepShape, DType.F32);
            SchedulerStepPacked(newLatent, velocityPred, packedLatent, scheduler, i);
            velocityPred.Dispose();
            packedLatent.Dispose();
            packedLatent = newLatent;

            LogPerLatentChannelMeanPacked($"Step {i+1} latent", packedLatent);

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        clipPooled.Dispose();
        t5Embeddings.Dispose();

        // Tear down the streaming controller (frees still-resident blocks) and free the
        // shared weights. After this, the transformer holds no device memory, making
        // room for VAE decode on tight VRAM budgets. The streaming + eager paths both
        // converge on the same final state: zero transformer weights resident.
        _transformer.BeforeBlockForward = null;
        if (streamer is not null)
        {
            streamer.EvictAll();
            streamer.Dispose();
            _backend.FreeWeights(_transformer.EnumerateSharedWeights());
        }
        else
        {
            _backend.FreeWeights(_transformer.EnumerateWeights());
        }

        // ── 5. Unpack latent: [1, seqLen, 64] → [1, 16, latentH, latentW] ──
        LogTensorStats("Final packed latent", packedLatent);
        Tensor unpackedLatent = UnpackLatent(packedLatent, latentH, latentW);
        packedLatent.Dispose();
        LogTensorStats("Unpacked latent", unpackedLatent);
        LogPerChannelStats("Unpacked latent", unpackedLatent);

        // Save unpacked latent for Python cross-validation if output dir exists
        string? debugLatentDir = Environment.GetEnvironmentVariable("FLUX_DEBUG_DIR");
        if (debugLatentDir is not null)
        {
            Directory.CreateDirectory(debugLatentDir);
            string latentPath = Path.Combine(debugLatentDir, "unpacked_latent.bin");
            ReadOnlySpan<float> latentData = unpackedLatent.AsReadOnlySpan<float>();
            using FileStream fs = new FileStream(latentPath, FileMode.Create);
            using BinaryWriter bw = new BinaryWriter(fs);
            for (int i = 0; i < latentData.Length; i++)
                bw.Write(latentData[i]);
            Logs.Info($"Saved unpacked latent to {latentPath} ({latentData.Length} floats, shape={unpackedLatent.Shape})");
        }

        // ── 6. VAE decode ─────────────────────────────────────────────
        // Preload VAE weights too — many small conv+norm ops in a row, same per-op
        // re-upload pattern would apply.
        _backend.PreloadWeights(_vaeDecoder.EnumerateWeights());

        // Always go through DecodeTiled. It internally fast-paths to a single direct
        // decode when the latent fits in one tile (small images), so there's no overhead
        // for normal sizes — but for anything ≥1024² it slices into 64-latent / 512-RGB
        // tiles with a 64-pixel RGB blend overlap. This kills the catastrophic im2col
        // workspace blow-up: a 256ch 3×3 conv at 1024² needs ~9.7 GB workspace as one
        // shot vs ~2.4 GB per tile (F32), regardless of final resolution.
        // (Tiles run at F32; F16 VAE produced black output on Flux Schnell — needs
        // GroupNorm/softmax precision investigation before re-enabling.)
        Logs.Info("Decoding latents to image (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(_backend, unpackedLatent);
        unpackedLatent.Dispose();
        LogTensorStats("VAE output", image);
        LogPerChannelStats("VAE output", image);
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // ── 7. Convert to RGB bytes ───────────────────────────────────
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Flux ({baseMode}) {opMode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, request.Width, request.Height, seed);
    }

    /// <summary>Picks <c>prefetchAhead</c> for the streaming controller based on how much VRAM
    /// is left after reserving for the peak activation working set of a single forward pass.
    /// Each extra prefetched block costs one block's worth of resident weights in addition to
    /// the current+next block already in flight. We pick the largest value that still leaves
    /// headroom for activations + cuBLAS workspace, capped at 2 (deeper just churns VRAM
    /// without extra hiding when blocks compute in tens of milliseconds).</summary>
    private static int ChooseFluxPrefetchAhead(
        IStreamingWeightCache cache,
        IStreamingBlock[] blocks,
        int txtSeqLen, int imgSeqLen, int hiddenSize, int mlpDim)
    {
        // Estimate the peak activation footprint of a single SingleStreamBlock — the deepest
        // valley in the forward pass, since SingleStreamBlocks hold both the F16 mlpInput +
        // mlpActivated and the F16 concatted simultaneously alongside the F32 attention
        // tensors. Numbers below are byte sizes for B=1.
        long totalSeqLen = txtSeqLen + imgSeqLen;
        long mlpInputBytes = totalSeqLen * mlpDim * 2;             // F16
        long mlpActivatedBytes = totalSeqLen * mlpDim * 2;         // F16
        long concattedBytes = totalSeqLen * (hiddenSize + mlpDim) * 2; // F16
        long attnFlatBytes = totalSeqLen * hiddenSize * 4;         // F32
        long qkvBytes = 3L * totalSeqLen * hiddenSize * 4;         // F32 q/k/v
        long mhBytes = 3L * totalSeqLen * hiddenSize * 4;          // F32 multi-head q/k/v
        long modulatedBytes = totalSeqLen * hiddenSize * 4;        // F32 modulated/normed
        long xBytes = totalSeqLen * hiddenSize * 4;                // F32 block input
        long scratchBytes = mlpInputBytes + mlpActivatedBytes + concattedBytes
            + attnFlatBytes + qkvBytes + mhBytes + modulatedBytes + xBytes;
        // Add a generous fudge for cuBLAS workspace, fp8→F16 weight cast buffers, RoPE tables,
        // and the persistent shared weights that aren't part of the streamed set. These are
        // hard to predict tightly; 1 GB is roomy enough to keep us out of OOM territory and
        // cheap if we have it.
        long activationReserve = scratchBytes + 1024L * 1024 * 1024;

        long avail = cache.QueryAvailableWeightCacheBytes(activationReserve);
        if (avail <= 0) return 0;

        long perBlockBytes = blocks.Length > 0 ? blocks[0].EstimatedWeightBytes : 0;
        if (perBlockBytes <= 0) return 1;

        // The streaming working set briefly hits (prefetchAhead + 2) blocks at the moment we
        // make block N+1 resident before evicting block N-1 (see BlockStreamingController.
        // BeforeBlockForward). Pick the largest prefetch that keeps that peak under budget.
        // Cap at 2 — beyond that we burn VRAM without much extra latency hiding.
        int maxByBudget = (int)(avail / perBlockBytes) - 2;
        int chosen = Math.Clamp(maxByBudget, 0, 2);
        return chosen;
    }

    /// <summary>Builds the initial packed latent for Flux denoising. T2I: fresh Gaussian noise scaled by the scheduler's initial sigma. Img2img: VAE-encoded source latent (16 channels) packed via <see cref="PackLatent"/>, combined with fresh packed noise via flow-matching <c>AddNoise</c>: <c>noisy = (1-sigma) * source + sigma * noise</c>.</summary>
    private Tensor BuildInitialPackedLatent(
        TextToImageRequest request,
        FlowMatchEulerDiscreteScheduler scheduler,
        TensorShape latentShape,
        TensorShape packedShape,
        int latentH, int latentW, int seed, int startStep)
    {
        Tensor packedNoise = PackLatent(SeedGenerator.CreateNoise(latentShape, seed), latentH, latentW);

        if (request is ImageToImageRequest img2img)
        {
            Stopwatch vaeEncSw = Stopwatch.StartNew();
            Tensor sourceUnpacked = _vaeEncoder!.Encode(_backend, img2img.SourceImage);
            vaeEncSw.Stop();
            Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

            Tensor sourcePacked = PackLatent(sourceUnpacked, latentH, latentW);
            sourceUnpacked.Dispose();

            Tensor result = new Tensor(packedShape, DType.F32);
            scheduler.AddNoise(result, sourcePacked, packedNoise, startStep);
            sourcePacked.Dispose();
            packedNoise.Dispose();
            return result;
        }

        // T2I path: scale packed noise by initSigma.
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(packedShape, DType.F32);
            _backend.Scale(scaled, packedNoise, initSigma);
            packedNoise.Dispose();
            return scaled;
        }
        return packedNoise;
    }

    private static void LogTensorStats(string name, Tensor tensor)
    {
        ReadOnlySpan<float> data = tensor.AsReadOnlySpan<float>();
        float min = float.MaxValue, max = float.MinValue, sum = 0;
        int nanCount = 0, infCount = 0, zeroCount = 0;
        for (int i = 0; i < data.Length; i++)
        {
            float v = data[i];
            if (float.IsNaN(v)) { nanCount++; continue; }
            if (float.IsInfinity(v)) { infCount++; continue; }
            if (v == 0f) zeroCount++;
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
        }
        float mean = data.Length > 0 ? sum / data.Length : 0;
        Logs.Debug($"  [{name}] shape={tensor.Shape} dtype={tensor.DType} min={min:E3} max={max:E3} mean={mean:E3} nan={nanCount} inf={infCount} zero={zeroCount}/{data.Length}");
    }

    /// <summary>Logs per-latent-channel mean for a packed [B, S, 64] tensor (velocity or latent). Each latent channel occupies 4 contiguous slots in the feature dim (c*4 .. c*4+3 for 2x2 patch).</summary>
    private static void LogPerLatentChannelMeanPacked(string name, Tensor packed)
    {
        long batch = packed.Shape[0];
        long seqLen = packed.Shape[1];
        long featDim = packed.Shape[2];
        int latentChannels = (int)(featDim / 4);
        float* ptr = (float*)packed.DataPointer;
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        sb.Append($"  [{name}] per-lat-ch mean: ");
        for (int c = 0; c < latentChannels; c++)
        {
            double sum = 0;
            long count = 0;
            for (long b = 0; b < batch; b++)
            {
                for (long s = 0; s < seqLen; s++)
                {
                    long baseIdx = (b * seqLen + s) * featDim + c * 4;
                    sum += ptr[baseIdx + 0];
                    sum += ptr[baseIdx + 1];
                    sum += ptr[baseIdx + 2];
                    sum += ptr[baseIdx + 3];
                    count += 4;
                }
            }
            float mean = (float)(sum / count);
            sb.Append($"c{c}={mean:+0.000;-0.000} ");
        }
        Logs.Debug(sb.ToString());
    }

    /// <summary>Logs per-channel statistics for a 4D NCHW tensor. Useful for diagnosing color channel imbalances.</summary>
    private static void LogPerChannelStats(string name, Tensor tensor)
    {
        int channels = (int)tensor.Shape[1];
        int spatial = (int)(tensor.Shape[2] * tensor.Shape[3]);
        float* ptr = (float*)tensor.DataPointer;
        for (int c = 0; c < channels; c++)
        {
            float cMin = float.MaxValue, cMax = float.MinValue, cSum = 0;
            for (int i = 0; i < spatial; i++)
            {
                float v = ptr[c * spatial + i];
                if (v < cMin) cMin = v;
                if (v > cMax) cMax = v;
                cSum += v;
            }
            float cMean = spatial > 0 ? cSum / spatial : 0;
            Logs.Info($"  [{name}] ch{c}: min={cMin:F4} max={cMax:F4} mean={cMean:F4}");
        }
    }

    /// <summary>Packs a latent tensor from [B, C, H, W] to [B, H/2*W/2, C*4]. Rearranges 2x2 spatial patches into channel dimension.</summary>
    private static Tensor PackLatent(Tensor latent, int h, int w)
    {
        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int hPacked = h / 2;
        int wPacked = w / 2;
        int patchDim = channels * 4; // 16 * 4 = 64
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
                    int outBase = (b * seqLen + seqIdx) * patchDim;

                    // For each channel, gather 2x2 patch
                    for (int c = 0; c < channels; c++)
                    {
                        int inChannelBase = (b * channels + c) * h * w;
                        int patchBase = outBase + c * 4;

                        outPtr[patchBase + 0] = inPtr[inChannelBase + (ph * 2 + 0) * w + (pw * 2 + 0)];
                        outPtr[patchBase + 1] = inPtr[inChannelBase + (ph * 2 + 0) * w + (pw * 2 + 1)];
                        outPtr[patchBase + 2] = inPtr[inChannelBase + (ph * 2 + 1) * w + (pw * 2 + 0)];
                        outPtr[patchBase + 3] = inPtr[inChannelBase + (ph * 2 + 1) * w + (pw * 2 + 1)];
                    }
                }
            }
        }

        return packed;
    }

    /// <summary>Unpacks a latent tensor from [B, H/2*W/2, C*4] back to [B, C, H, W].</summary>
    private static Tensor UnpackLatent(Tensor packed, int h, int w)
    {
        int batch = (int)packed.Shape[0];
        int channels = 16; // Flux always uses 16 latent channels
        int hPacked = h / 2;
        int wPacked = w / 2;
        int patchDim = channels * 4;
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
                    int inBase = (b * seqLen + seqIdx) * patchDim;

                    for (int c = 0; c < channels; c++)
                    {
                        int outChannelBase = (b * channels + c) * h * w;
                        int patchBase = inBase + c * 4;

                        outPtr[outChannelBase + (ph * 2 + 0) * w + (pw * 2 + 0)] = inPtr[patchBase + 0];
                        outPtr[outChannelBase + (ph * 2 + 0) * w + (pw * 2 + 1)] = inPtr[patchBase + 1];
                        outPtr[outChannelBase + (ph * 2 + 1) * w + (pw * 2 + 0)] = inPtr[patchBase + 2];
                        outPtr[outChannelBase + (ph * 2 + 1) * w + (pw * 2 + 1)] = inPtr[patchBase + 3];
                    }
                }
            }
        }

        return unpacked;
    }

    /// <summary>Performs a scheduler step on packed latent tokens. Euler: x_next = x + v * dt, where dt = sigma_next - sigma.</summary>
    private static void SchedulerStepPacked(Tensor output, Tensor velocity, Tensor sample,
        FlowMatchEulerDiscreteScheduler scheduler, int stepIndex)
    {
        // The scheduler operates on flat data regardless of shape
        scheduler.Step(output, velocity, sample, stepIndex);
    }

    /// <summary>Extracts the hidden state at the EOS token position from [B, seqLen, hiddenSize] to [B, hiddenSize]. Used to get CLIP-L pooled output for Flux (no text_projection needed).</summary>
    private static Tensor ExtractEosHiddenState(Tensor hidden, int eosPosition)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];
        int hiddenSize = (int)hidden.Shape[2];

        TensorShape pooledShape = new TensorShape(batch, hiddenSize);
        Tensor pooled = new Tensor(pooledShape, DType.F32);

        float* srcPtr = (float*)hidden.DataPointer;
        float* dstPtr = (float*)pooled.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int srcOffset = (b * seqLen + eosPosition) * hiddenSize;
            int dstOffset = b * hiddenSize;
            for (int d = 0; d < hiddenSize; d++)
            {
                dstPtr[dstOffset + d] = srcPtr[srcOffset + d];
            }
        }

        return pooled;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    /// <summary>Disposes the pipeline. Does not dispose the backend or model components (shared resources).</summary>
    public void Dispose()
    {
        Volatile.Write(ref _disposed, 1);
    }
}
