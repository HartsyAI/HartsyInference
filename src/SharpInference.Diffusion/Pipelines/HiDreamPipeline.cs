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
public sealed unsafe class HiDreamPipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly ClipTextEncoder _clipL;
    private readonly ClipTextEncoder _clipG;
    private readonly T5TextEncoder _t5;
    private readonly LlamaStyleEncoder _llama;
    private readonly HiDreamTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly HiDreamConfig _config;
    private int _disposed;

    /// <summary>Creates a new HiDream pipeline with all components pre-loaded. Caller owns each component
    /// and is responsible for their lifetime — the pipeline does not dispose them on its own
    /// <see cref="Dispose"/>.</summary>
    public HiDreamPipeline(IBackend backend,
        ClipTextEncoder clipL, ClipTextEncoder clipG,
        T5TextEncoder t5, LlamaStyleEncoder llama,
        HiDreamTransformer transformer, VaeDecoder vaeDecoder,
        HiDreamConfig config)
    {
        _backend = backend;
        _clipL = clipL;
        _clipG = clipG;
        _t5 = t5;
        _llama = llama;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-tokenized inputs for all four text encoders.</summary>
    /// <param name="promptTokenIdsL">Prompt token IDs for CLIP-L [seqLen].</param>
    /// <param name="negativePromptTokenIdsL">Negative prompt token IDs for CLIP-L. Used only when <c>CfgScale &gt; 1</c>.</param>
    /// <param name="promptTokenIdsG">Prompt token IDs for CLIP-G [seqLen].</param>
    /// <param name="negativePromptTokenIdsG">Negative prompt token IDs for CLIP-G.</param>
    /// <param name="promptEosPositionL">EOS position in prompt for CLIP-L (for pooled output).</param>
    /// <param name="negativeEosPositionL">EOS position in negative prompt for CLIP-L.</param>
    /// <param name="promptEosPositionG">EOS position in prompt for CLIP-G.</param>
    /// <param name="negativeEosPositionG">EOS position in negative prompt for CLIP-G.</param>
    /// <param name="promptTokenIdsT5">Prompt token IDs for T5-XXL [seqLen].</param>
    /// <param name="negativePromptTokenIdsT5">Negative prompt token IDs for T5-XXL.</param>
    /// <param name="promptAttentionMaskT5">T5 attention mask for prompt (1=attend, 0=pad). Null = attend all.</param>
    /// <param name="negativeAttentionMaskT5">T5 attention mask for negative prompt.</param>
    /// <param name="promptTokenIdsLlama">Prompt token IDs for Llama-3.1 [seqLen].</param>
    /// <param name="negativePromptTokenIdsLlama">Negative prompt token IDs for Llama-3.1.</param>
    /// <param name="request">Generation parameters.</param>
    /// <param name="onProgress">Optional per-step progress callback.</param>
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
        int width = request.Width;
        int height = request.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;
        bool useCfg = cfgScale > 1.0f;

        Logs.Info($"HiDream t2i: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode all four text encoders (positive + optional negative) ──
        Logs.Info("Encoding text with CLIP-L, CLIP-G, T5-XXL, and Llama-3.1...");

        (Tensor condPooled, Tensor condT5, IReadOnlyList<Tensor> condLlama) = EncodePrompt(
            promptTokenIdsL, promptTokenIdsG, promptTokenIdsT5, promptTokenIdsLlama,
            promptEosPositionL, promptEosPositionG, promptAttentionMaskT5);

        Tensor? uncondPooled = null;
        Tensor? uncondT5 = null;
        IReadOnlyList<Tensor>? uncondLlama = null;
        if (useCfg)
        {
            (uncondPooled, uncondT5, uncondLlama) = EncodePrompt(
                negativePromptTokenIdsL, negativePromptTokenIdsG, negativePromptTokenIdsT5, negativePromptTokenIdsLlama,
                negativeEosPositionL, negativeEosPositionG, negativeAttentionMaskT5);
        }

        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

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
            _backend.Scale(scaled, latent, initSigma);
            latent.Dispose();
            latent = scaled;
        }

        // ── 4. Denoising loop ──
        Logs.Info("Starting HiDream denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor noisePred;
            if (useCfg)
            {
                Tensor condNoise = _transformer.Forward(_backend, latent, t, condT5, condLlama, condPooled);
                Tensor uncondNoise = _transformer.Forward(_backend, latent, t, uncondT5!, uncondLlama!, uncondPooled!);
                noisePred = ApplyCfg(latent.Shape, uncondNoise, condNoise, cfgScale);
                condNoise.Dispose();
                uncondNoise.Dispose();
            }
            else
            {
                noisePred = _transformer.Forward(_backend, latent, t, condT5, condLlama, condPooled);
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

        condPooled.Dispose();
        condT5.Dispose();
        for (int i = 0; i < condLlama.Count; i++) condLlama[i].Dispose();
        uncondPooled?.Dispose();
        uncondT5?.Dispose();
        if (uncondLlama is not null)
            for (int i = 0; i < uncondLlama.Count; i++) uncondLlama[i].Dispose();

        HiDreamTransformer.DumpFinalLatent(latent);

        // Free transformer + text-encoder weights from VRAM before VAE decode (Phase 3 deviations #18).
        _backend.Sync();
        _backend.FreeWeights(_transformer.EnumerateWeights());
        _backend.FreeWeights(_t5.EnumerateWeights());
        _backend.FreeWeights(_llama.EnumerateWeights());

        // ── 5. VAE decode ──
        Logs.Verbose("Decoding latents to image (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(_backend, latent);
        latent.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"HiDream t2i complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (rgbData, width, height, seed);
    }

    /// <summary>Encodes a single prompt through all four text encoders, returning (pooled, t5_hidden, llama_per_block_hidden).</summary>
    private (Tensor pooled, Tensor t5Hidden, IReadOnlyList<Tensor> llamaPerBlock) EncodePrompt(
        int[] tokenIdsL, int[] tokenIdsG, int[] tokenIdsT5, int[] tokenIdsLlama,
        int eosPositionL, int eosPositionG, int[]? attentionMaskT5)
    {
        // CLIP-L pooled
        int[][] batchL = [tokenIdsL];
        int[] eosL = [eosPositionL];
        (Tensor _, Tensor? clipLPooled) = _clipL.EncodePenultimate(_backend, batchL, eosL);

        // CLIP-G pooled
        int[][] batchG = [tokenIdsG];
        int[] eosG = [eosPositionG];
        (Tensor _, Tensor? clipGPooled) = _clipG.EncodePenultimate(_backend, batchG, eosG);

        // Pooled = concat(clipL_pooled, clipG_pooled, dim=-1) -> [B, 2048]
        Tensor pooled = ConcatPooled(clipLPooled!, clipGPooled!);
        clipLPooled?.Dispose();
        clipGPooled?.Dispose();

        // T5-XXL hidden [B, S_t5, 4096]
        int[][] batchT5 = [tokenIdsT5];
        int[][]? batchMask = attentionMaskT5 is not null ? [attentionMaskT5] : null;
        Tensor t5Hidden = _t5.Encode(_backend, batchT5, batchMask);

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

        Tensor stacked = _llama.EncodeMultiLayer(_backend, batchLlama, uniqueArray);
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

    /// <summary>CFG combine: out = uncond + cfgScale * (cond - uncond).</summary>
    private static Tensor ApplyCfg(TensorShape shape, Tensor uncond, Tensor cond, float cfgScale)
    {
        Tensor output = new Tensor(shape, DType.F32);
        float* up = (float*)uncond.DataPointer;
        float* cp = (float*)cond.DataPointer;
        float* op = (float*)output.DataPointer;
        long count = shape.ElementCount;
        for (long i = 0; i < count; i++)
            op[i] = up[i] + cfgScale * (cp[i] - up[i]);
        return output;
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

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    /// <summary>Disposes the pipeline. Does not dispose the backend or model components (shared resources).</summary>
    public void Dispose()
    {
        Volatile.Write(ref _disposed, 1);
    }
}
