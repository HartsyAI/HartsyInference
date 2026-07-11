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

/// <summary>HiDream-I1 text-to-image pipeline. Orchestrates a quad text-encoder stack
/// (CLIP-L + CLIP-G → pooled 2048-d, T5-XXL → 4096-d sequence, Llama-3.1 → multi-layer 4096-d hidden states),
/// runs the <see cref="HiDreamTransformer"/> with flow-match Euler scheduling, and decodes the predicted
/// velocity through the 16-channel VAE.
/// <para>This pipeline is text-to-image only — image-to-image and editing pipelines are not provided.</para>
/// <para>The HiDream transformer consumes one Llama hidden state per block (16 + 32 = 48 layers per the
/// default config). The pipeline drives <see cref="LlamaStyleEncoder.EncodeMultiLayer"/> with the layer
/// indices configured in <see cref="HiDreamConfig.LlamaLayers"/>; the encoder returns a single packed
/// tensor with all selected layers concatenated along the last dim, which we slice into per-layer tensors
/// before calling the transformer.</para></summary>
public sealed unsafe class HiDreamPipeline : DiffusionPipelineBase
{
    private readonly ClipTextEncoder _clipL;
    private readonly ClipTextEncoder _clipG;
    private readonly T5TextEncoder _t5;
    private readonly LlamaStyleEncoder _llama;
    private readonly HiDreamTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly HiDreamConfig _config;

    /// <summary>Keeps the 17 GB fp8 DiT GPU-resident across generations (skips the post-loop FreeWeights +
    /// next-gen ~5 s re-upload). The quad encoder stack (T5 ~5 GB + Llama ~8 GB) cannot co-reside with it,
    /// so a prompt-cache MISS under this flag frees the DiT first, encodes, then re-preloads — repeat
    /// prompts skip both. Standard-profile default ON (HARTSY_KEEP_MODELS=0 disables).</summary>
    private static readonly bool KeepModelsResident =
        HartsyInference.Core.Runtime.EnvSwitch.IsEnabled("HARTSY_KEEP_MODELS", defaultOn: true);
    private bool _ditResident;

    // Prompt-conditioning cache (one cond + one uncond slot): the quad encode (CLIP-L/G + T5 + Llama-8B,
    // 49 per-block hidden states) costs seconds per generation and its outputs are pure functions of the
    // token ids. Keyed on (T5 ids, Llama ids) — the four tokenizations covary with the prompt string, and
    // T5+Llama together pin it. Tensors are host-materialized at store time so activation reclaims can't
    // revert them; the forward must NOT dispose cache-owned tensors.
    private int[]? _cachedCondKeyT5, _cachedCondKeyLlama;
    private Tensor? _cachedCondPooled, _cachedCondT5;
    private IReadOnlyList<Tensor>? _cachedCondLlama;
    private int[]? _cachedUncondKeyT5, _cachedUncondKeyLlama;
    private Tensor? _cachedUncondPooled, _cachedUncondT5;
    private IReadOnlyList<Tensor>? _cachedUncondLlama;

    /// <summary>Creates a new HiDream pipeline with all components pre-loaded. Caller owns each component
    /// and is responsible for their lifetime — the pipeline does not dispose them on its own
    /// <see cref="DiffusionPipelineBase.Dispose"/>.</summary>
    public HiDreamPipeline(IBackend backend,
        ClipTextEncoder clipL, ClipTextEncoder clipG,
        T5TextEncoder t5, LlamaStyleEncoder llama,
        HiDreamTransformer transformer, VaeDecoder vaeDecoder,
        HiDreamConfig config)
        : base(backend)
    {
        _clipL = clipL;
        _clipG = clipG;
        _t5 = t5;
        _llama = llama;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-tokenized inputs for all four text encoders.</summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsL, int[] negativePromptTokenIdsL,
        int[] promptTokenIdsG, int[] negativePromptTokenIdsG,
        int promptEosPositionL, int negativeEosPositionL,
        int promptEosPositionG, int negativeEosPositionG,
        int[] promptTokenIdsT5, int[] negativePromptTokenIdsT5,
        int[]? promptAttentionMaskT5, int[]? negativeAttentionMaskT5,
        int[] promptTokenIdsLlama, int[] negativePromptTokenIdsLlama,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, float cfgScale, int width, int height) = GenerationDefaults.HiDreamFull.Resolve(request);
        int latentH = height / 8;
        int latentW = width / 8;
        bool useCfg = cfgScale > 1.0f;

        Logs.Info($"HiDream t2i: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode all four text encoders (positive + optional negative), with a prompt cache ──
        bool condHit = _cachedCondPooled is not null
            && _cachedCondKeyT5 is not null && _cachedCondKeyT5.AsSpan().SequenceEqual(promptTokenIdsT5)
            && _cachedCondKeyLlama is not null && _cachedCondKeyLlama.AsSpan().SequenceEqual(promptTokenIdsLlama);
        bool uncondHit = !useCfg || (_cachedUncondPooled is not null
            && _cachedUncondKeyT5 is not null && _cachedUncondKeyT5.AsSpan().SequenceEqual(negativePromptTokenIdsT5)
            && _cachedUncondKeyLlama is not null && _cachedUncondKeyLlama.AsSpan().SequenceEqual(negativePromptTokenIdsLlama));

        Tensor condPooled, condT5;
        IReadOnlyList<Tensor> condLlama;
        Tensor? uncondPooled = null;
        Tensor? uncondT5 = null;
        IReadOnlyList<Tensor>? uncondLlama = null;
        if (condHit && uncondHit)
        {
            condPooled = _cachedCondPooled!;
            condT5 = _cachedCondT5!;
            condLlama = _cachedCondLlama!;
            if (useCfg)
            {
                uncondPooled = _cachedUncondPooled;
                uncondT5 = _cachedUncondT5;
                uncondLlama = _cachedUncondLlama;
            }
            Logs.Info("[HiDream] prompt-conditioning cache hit — quad-encoder phase skipped");
        }
        else
        {
            Logs.Info("Encoding text with CLIP-L, CLIP-G, T5-XXL, and Llama-3.1...");
            if (_ditResident)
            {
                // The quad encoders cannot co-reside with the resident 17 GB DiT; evict for this
                // new-prompt generation and re-preload below.
                Backend.Sync();
                Backend.FreeWeights(_transformer.EnumerateWeights());
                _ditResident = false;
            }

            // Bulk-upload T5 + Llama-3.1 weights once. These are the heavy encoders (T5 ~5 GB, Llama
            // ~8 GB) and the kernels inside touch them on every layer — per-op cache misses would
            // dominate first-generation latency. CLIP-L/G are tiny and stay lazy-cached. Paired
            // with FreeWeights below. No-op on backends without a weight cache.
            Backend.PreloadWeights(_t5.EnumerateWeights());
            Backend.PreloadWeights(_llama.EnumerateWeights());

            (condPooled, condT5, condLlama) = EncodePrompt(
                promptTokenIdsL, promptTokenIdsG, promptTokenIdsT5, promptTokenIdsLlama,
                promptEosPositionL, promptEosPositionG, promptAttentionMaskT5);

            if (useCfg)
            {
                (uncondPooled, uncondT5, uncondLlama) = EncodePrompt(
                    negativePromptTokenIdsL, negativePromptTokenIdsG, negativePromptTokenIdsT5, negativePromptTokenIdsLlama,
                    negativeEosPositionL, negativeEosPositionG, negativeAttentionMaskT5);
            }

            Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

            // Free the text-encoder weights from VRAM now that conditioning is computed — the 17 GB fp8
            // transformer must fit the 4090 alone (T5 ~5 GB + Llama ~8 GB would not co-reside with it). The
            // computed conditioning tensors (condT5/condLlama/condPooled) are separate activations and survive.
            Backend.Sync();
            Backend.FreeWeights(_clipL.EnumerateWeights());
            Backend.FreeWeights(_clipG.EnumerateWeights());
            Backend.FreeWeights(_t5.EnumerateWeights());
            Backend.FreeWeights(_llama.EnumerateWeights());
            // Return the encode phase's pool reserve to the driver before the 17 GB DiT preload: under the
            // warm-pool profile the Llama/T5 encode leaves multi-GB reservations that would otherwise starve
            // the CFG denoise loop (24 GB − 17 GB weights leaves ~7 GB for everything else — measured OOM on
            // step 1 without this planned trim).
            Backend.TrimMemoryPool();

            // Host-materialize the conditioning so it survives activation reclaims, then cache it.
            _ = condPooled.DataPointer;
            _ = condT5.DataPointer;
            foreach (Tensor t in condLlama) _ = t.DataPointer;
            DisposeCachedCond();
            _cachedCondPooled = condPooled;
            _cachedCondT5 = condT5;
            _cachedCondLlama = condLlama;
            _cachedCondKeyT5 = (int[])promptTokenIdsT5.Clone();
            _cachedCondKeyLlama = (int[])promptTokenIdsLlama.Clone();
            if (useCfg)
            {
                _ = uncondPooled!.DataPointer;
                _ = uncondT5!.DataPointer;
                foreach (Tensor t in uncondLlama!) _ = t.DataPointer;
                DisposeCachedUncond();
                _cachedUncondPooled = uncondPooled;
                _cachedUncondT5 = uncondT5;
                _cachedUncondLlama = uncondLlama;
                _cachedUncondKeyT5 = (int[])negativePromptTokenIdsT5.Clone();
                _cachedUncondKeyLlama = (int[])negativePromptTokenIdsLlama.Clone();
            }
        }

        // ── 2. Set up flow-match scheduler ──
        TensorShape latentShape = new TensorShape(1, _config.InChannels, latentH, latentW);
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(_config.SchedulerShift);
        scheduler.SetTimesteps(steps);

        // ── 3. Initial noise latent ──
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(latentShape, DType.F32);
            Backend.Scale(scaled, latent, initSigma);
            latent.Dispose();
            latent = scaled;
        }

        // ── 4. Denoising loop ──
        // Bulk-upload transformer weights before the denoise loop (no-op when already resident under
        // HARTSY_KEEP_MODELS). Paired with the conditional FreeWeights at the VAE handoff.
        Backend.PreloadWeights(_transformer.EnumerateWeights());
        _ditResident = true;

        Logs.Info("Starting HiDream denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor noisePred;
            if (useCfg)
            {
                Tensor condNoise = _transformer.Forward(Backend, latent, t, condT5, condLlama, condPooled);
                // Bound the async queue to one forward's transients: with the blocks now fully GPU-resident
                // (no implicit host-sync throttling), two queued 17B F32-activation forwards overflow the
                // ~7 GB left beside the resident fp8 weights (measured step-1 OOM at 1024²-CFG). One forward
                // fits; serialize the pair.
                Backend.Sync();
                Tensor uncondNoise = _transformer.Forward(Backend, latent, t, uncondT5!, uncondLlama!, uncondPooled!);
                noisePred = CfgHelper.ApplyCfg(uncondNoise, condNoise, cfgScale);
                condNoise.Dispose();
                uncondNoise.Dispose();
            }
            else
            {
                noisePred = _transformer.Forward(Backend, latent, t, condT5, condLlama, condPooled);
            }

            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePred, latent, i);
            noisePred.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        // Conditioning tensors are cross-generation cache-owned — NOT disposed here.

        HiDreamTransformer.DumpFinalLatent(latent);

        // Under HARTSY_KEEP_MODELS the 17 GB fp8 DiT stays resident (the tiled VAE decode fits beside it);
        // otherwise free it before the decode as before. Phase 3 deviations #18.
        Backend.Sync();
        if (!KeepModelsResident)
        {
            Backend.FreeWeights(_transformer.EnumerateWeights());
            _ditResident = false;
        }
        else
        {
            // DiT stays resident: hand the loop's pool reserve back so the VAE decode's im2col bands fit
            // beside the 17 GB weights.
            Backend.TrimMemoryPool();
        }

        // ── 5. VAE decode ──
        Logs.Verbose("Decoding latents to image (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        // 48-latent tiles instead of the 64 default: at 128×128 latent both give a 3×3 grid, but the
        // smaller tile drops per-tile conv activations ~44% — the decode beside the resident 17 GB DiT
        // was the whole generation's VRAM peak (measured 23.9 GB with 64, ~620 MB headroom on a 24 GB card).
        Tensor image = _vaeDecoder.DecodeTiled(Backend, latent, tileLatentSize: 48);
        latent.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        // The full-res decode's im2col bands leave a ~3 GB pool reservation on top of the resident 17 GB
        // DiT (measured 23.5 GB retained watermark); hand it back so follow-up work — or another model's
        // load — starts from the ~20 GB loop plateau instead of the decode peak.
        if (KeepModelsResident)
            Backend.TrimMemoryPool();

        sw.Stop();
        Logs.Info($"HiDream t2i complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (rgbData, width, height, seed);
    }

    /// <summary>Disposes the cached positive conditioning (safe mid-session — the context is live at
    /// replacement time; end-of-life teardown nulls via the weight-field pattern instead).</summary>
    private void DisposeCachedCond()
    {
        _cachedCondPooled?.Dispose();
        _cachedCondT5?.Dispose();
        if (_cachedCondLlama is not null)
            foreach (Tensor t in _cachedCondLlama) t.Dispose();
        _cachedCondPooled = null;
        _cachedCondT5 = null;
        _cachedCondLlama = null;
    }

    /// <summary>Negative twin of <see cref="DisposeCachedCond"/>.</summary>
    private void DisposeCachedUncond()
    {
        _cachedUncondPooled?.Dispose();
        _cachedUncondT5?.Dispose();
        if (_cachedUncondLlama is not null)
            foreach (Tensor t in _cachedUncondLlama) t.Dispose();
        _cachedUncondPooled = null;
        _cachedUncondT5 = null;
        _cachedUncondLlama = null;
    }

    /// <summary>Encodes a single prompt through all four text encoders, returning (pooled, t5_hidden, llama_per_block_hidden).</summary>
    private (Tensor pooled, Tensor t5Hidden, IReadOnlyList<Tensor> llamaPerBlock) EncodePrompt(
        int[] tokenIdsL, int[] tokenIdsG, int[] tokenIdsT5, int[] tokenIdsLlama,
        int eosPositionL, int eosPositionG, int[]? attentionMaskT5)
    {
        // CLIP-L pooled
        int[][] batchL = [tokenIdsL];
        int[] eosL = [eosPositionL];
        (Tensor _, Tensor? clipLPooled) = _clipL.EncodePenultimate(Backend, batchL, eosL);

        // CLIP-G pooled
        int[][] batchG = [tokenIdsG];
        int[] eosG = [eosPositionG];
        (Tensor _, Tensor? clipGPooled) = _clipG.EncodePenultimate(Backend, batchG, eosG);

        // Pooled = concat(clipL_pooled, clipG_pooled, dim=-1) -> [B, 2048]
        Tensor pooled = ConcatPooled(clipLPooled!, clipGPooled!);
        clipLPooled?.Dispose();
        clipGPooled?.Dispose();

        // T5-XXL hidden [B, S_t5, 4096]
        int[][] batchT5 = [tokenIdsT5];
        int[][]? batchMask = attentionMaskT5 is not null ? [attentionMaskT5] : null;
        Tensor t5Hidden = _t5.Encode(Backend, batchT5, batchMask);

        // Llama multi-layer: extract each requested layer separately. The diffusers reference indexes
        // hidden_states[1:] (one per encoder layer, dropping the embeddings) and feeds those into
        // caption_projection[0]. We use EncodeMultiLayer with the layer indices from config.
        int[][] batchLlama = [tokenIdsLlama];
        int[] llamaIndices = _config.LlamaLayers;

        // Need a strictly ascending list for EncodeMultiLayer; the config has duplicates of layer 31
        // (the default LlamaLayers replicates 31 sixteen times to fill 48 entries). We deduplicate
        // before the encode and then duplicate the slices afterwards.
        SortedSet<int> uniqueLayers = new();
        for (int i = 0; i < llamaIndices.Length; i++) uniqueLayers.Add(llamaIndices[i]);
        int[] uniqueArray = uniqueLayers.ToArray();

        Tensor stacked = _llama.EncodeMultiLayer(Backend, batchLlama, uniqueArray);
        // stacked is [B, S, K * H_llama]. Split into K per-layer tensors of [B, S, H_llama], then expand
        // to one tensor per LlamaLayers entry by mapping each entry to its slot in uniqueArray.
        int H = (int)stacked.Shape[2] / uniqueArray.Length;
        Tensor[] uniquePerLayer = SliceLastDimIntoChunks(stacked, uniqueArray.Length, H);
        stacked.Dispose();

        Dictionary<int, Tensor> layerToTensor = new(uniqueArray.Length);
        for (int i = 0; i < uniqueArray.Length; i++) layerToTensor[uniqueArray[i]] = uniquePerLayer[i];

        // Build the per-block list, cloning when a layer index repeats.
        Tensor[] perBlock = new Tensor[llamaIndices.Length];
        HashSet<Tensor> claimed = new();
        for (int i = 0; i < llamaIndices.Length; i++)
        {
            Tensor src = layerToTensor[llamaIndices[i]];
            if (claimed.Add(src))
            {
                perBlock[i] = src;
            }
            else
            {
                perBlock[i] = CloneTensor(src);
            }
        }

        return (pooled, t5Hidden, perBlock);
    }

    /// <summary>Splits [B, S, K*H] into K [B, S, H] tensors along the last dim.</summary>
    private static Tensor[] SliceLastDimIntoChunks(Tensor input, int K, int H)
    {
        int batch = (int)input.Shape[0];
        int seqLen = (int)input.Shape[1];
        Tensor[] outputs = new Tensor[K];
        float* inPtr = (float*)input.DataPointer;

        for (int k = 0; k < K; k++)
        {
            TensorShape shape = new TensorShape(batch, seqLen, H);
            Tensor chunk = new Tensor(shape, DType.F32);
            float* outPtr = (float*)chunk.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    int srcOff = (b * seqLen + s) * (K * H) + k * H;
                    int dstOff = (b * seqLen + s) * H;
                    Buffer.MemoryCopy(inPtr + srcOff, outPtr + dstOff, H * sizeof(float), H * sizeof(float));
                }
            }
            outputs[k] = chunk;
        }
        return outputs;
    }

    /// <summary>Returns a freshly allocated F32 copy of the given tensor.</summary>
    private static Tensor CloneTensor(Tensor src)
    {
        Tensor dst = new Tensor(src.Shape, DType.F32);
        long bytes = src.Shape.ElementCount * sizeof(float);
        Buffer.MemoryCopy((float*)src.DataPointer, (float*)dst.DataPointer, bytes, bytes);
        return dst;
    }

    /// <summary>Concatenates two pooled tensors [B, D1] and [B, D2] along the last dim → [B, D1+D2].</summary>
    private static Tensor ConcatPooled(Tensor a, Tensor b)
    {
        int batch = (int)a.Shape[0];
        int dimA = (int)a.Shape[1];
        int dimB = (int)b.Shape[1];
        int dimOut = dimA + dimB;
        TensorShape outShape = new TensorShape(batch, dimOut);
        Tensor output = new Tensor(outShape, DType.F32);

        float* aPtr = (float*)a.DataPointer;
        float* bPtr = (float*)b.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            Buffer.MemoryCopy(aPtr + bIdx * dimA, outPtr + bIdx * dimOut, dimA * sizeof(float), dimA * sizeof(float));
            Buffer.MemoryCopy(bPtr + bIdx * dimB, outPtr + bIdx * dimOut + dimA, dimB * sizeof(float), dimB * sizeof(float));
        }
        return output;
    }
}
