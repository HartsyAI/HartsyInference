using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>HunyuanVideo (Tencent, 13B) text-to-video pipeline. Drives the <see cref="HunyuanVideoDit"/> with
/// flow-match Euler (shift 7.0) and <b>embedded guidance</b> — a single forward per step (no CFG), matching
/// diffusers <c>HunyuanVideoPipeline</c> defaults (<c>true_cfg_scale=1</c>). The 13B bf16 DiT (~24 GB) is streamed
/// block-by-block (<see cref="BlockStreamingController"/>) on backends that expose a streaming cache; the shared
/// embedders/refiner/final layer stay resident.
///
/// <para>Consumes pre-computed text conditioning: <paramref name="promptEmbeds"/> is the cropped Llava-Llama-3
/// hidden state <c>[1, L, 4096]</c> (layer −3, template preamble removed); <paramref name="pooled"/> is the CLIP-L
/// <c>pooler_output</c> <c>[1, 768]</c>. Encode + free the text encoders upstream (both can't be resident with the
/// DiT on 24 GB). Latent grid: <c>T_lat=(F−1)/4+1</c>, <c>H/8</c>, <c>W/8</c>, 16 channels; the DiT patchifies
/// <c>(1,2,2)</c> internally. Decode divides by the VAE scaling factor 0.476986 (no mean/std shift).
/// <b>Status: built, first-run numeric validation pending.</b></para></summary>
public sealed unsafe class HunyuanVideoPipeline(IBackend backend, HunyuanVideoDit dit, HunyuanVideoVaeDecoder vae,
    HunyuanVideoConfig config) : DiffusionPipelineBase(backend)
{
    private const int VaeSpatial = 8;
    private const int VaeTemporal = 4;
    private const float VaeScalingFactor = 0.476986f;

    private readonly HunyuanVideoDit _dit = dit;
    private readonly HunyuanVideoVaeDecoder _vae = vae;
    private readonly HunyuanVideoConfig _config = config;

    /// <summary>Generates frames from pre-computed text conditioning. Returns one interleaved-RGB <c>byte[]</c> per
    /// frame plus the resolved dimensions and seed.</summary>
    public (byte[][] frames, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor promptEmbeds, Tensor pooled, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress = null)
    {
        Tensor latent = RunDenoise(promptEmbeds, pooled, request, numFrames, onProgress, out int seed);
        Tensor rgb;
        try { rgb = DecodeLatent(latent); }
        finally { latent.Dispose(); }

        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = VideoRgbFrames.ExtractFrame(rgb, i);
        int w = (int)rgb.Shape[4], h = (int)rgb.Shape[3];
        rgb.Dispose();
        Logs.Info($"HunyuanVideo T2V complete ({frames.Length} frames, seed={seed})");
        return (frames, w, h, seed);
    }

    private Tensor DecodeLatent(Tensor latent)
    {
        float* lp = (float*)latent.DataPointer;
        long n = latent.Shape.ElementCount;
        float inv = 1f / VaeScalingFactor;
        for (long i = 0; i < n; i++) lp[i] *= inv;
        Backend.PreloadWeights(_vae.EnumerateWeights());
        // Full-res 3D decode peaks >24 GB at 512×320×25f; spatial tiling bounds it to a per-tile working set.
        Tensor rgb = _vae.DecodeTiled(Backend, latent);
        Backend.Sync();
        Backend.FreeWeights(_vae.EnumerateWeights());
        return rgb;
    }

    /// <summary>Flow-match Euler denoise; returns the VAE-ready latent <c>[1,16,T_lat,H_lat,W_lat]</c>.</summary>
    private Tensor RunDenoise(Tensor promptEmbeds, Tensor pooled, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress, out int seed)
    {
        ThrowIfDisposed();
        seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? 512, height = request.Height ?? 320;
        if (width % VaeSpatial != 0 || height % VaeSpatial != 0)
            throw new ArgumentException($"Width/height must be divisible by {VaeSpatial} for HunyuanVideo.");
        if (numFrames < 1 || (numFrames - 1) % VaeTemporal != 0)
            throw new ArgumentException($"num_frames must satisfy (num_frames-1) % {VaeTemporal} == 0; got {numFrames}.");

        int tLat = (numFrames - 1) / VaeTemporal + 1;
        int hLat = height / VaeSpatial, wLat = width / VaeSpatial;
        int steps = request.Steps ?? _config.InferSteps;
        float shift = _config.FlowShift;
        float guidance = _config.GuidanceEmbed ? _config.EmbeddedGuidanceScale * 1000f : 0f;

        Logs.Info($"HunyuanVideo T2V: {numFrames}f {width}x{height}, {steps} steps, embedded-guidance={_config.EmbeddedGuidanceScale}, " +
            $"seed={seed} (latent {tLat}x{hLat}x{wLat}, shift={shift:F2})");
        Logs.Warning("HunyuanVideo pipeline is first-run-validation pending — numerics unverified vs the reference checkpoint.");

        // Keep the DiT fully resident when it fits (fp8 ~13 GB on a 24 GB card — no per-step streaming tax);
        // stream the 60 double+single blocks only when it doesn't (24 GB bf16 DiT).
        long ditWeightBytes = 0;
        foreach (Tensor t in _dit.EnumerateWeights()) ditWeightBytes += t.Shape.ElementCount * t.DType.SizeInBytes;
        const long residentMarginBytes = 6L << 30;   // activations + workspace headroom for one forward
        bool resident = Backend.StreamingCache is null
            || ditWeightBytes + residentMarginBytes <= Backend.FreeMemoryBytes();
        BlockStreamingController? streamer = null;
        if (!resident)
        {
            Backend.PreloadWeights(_dit.EnumerateSharedWeights());
            IStreamingBlock[] blocks = new IStreamingBlock[_dit.BlockCount];
            for (int b = 0; b < blocks.Length; b++) blocks[b] = _dit.GetBlock(b);
            streamer = new BlockStreamingController(
                Backend.StreamingCache!, blocks, prefetchAhead: 2, retainBehind: 0, backend: Backend);
            _dit.BeforeBlockForward = streamer.BeforeBlockForward;
            streamer.Prime();
            Logs.Info($"HunyuanVideo streaming: {blocks.Length} blocks, ~{streamer.EstimatedTotalWeightBytes / (1024 * 1024)} MB total");
        }
        else
        {
            Backend.PreloadWeights(_dit.EnumerateWeights());
            Logs.Info($"HunyuanVideo DiT resident: {ditWeightBytes / (1024 * 1024)} MB preloaded (no block streaming).");
        }

        Tensor latent = SeedGenerator.CreateNoise(new TensorShape([1L, _config.OutChannels, tLat, hLat, wLat]), seed);
        float[] tsteps = LancePipelineCommon.BuildShiftedTimesteps(steps, shift);

        // Sampler selection (2026-08-20). HunyuanVideo had no user-selectable sampler: the loop below was a hardcoded
        // flow-match Euler over `tsteps`. That array IS the family's sigma schedule — descending from 1.0 with an exact
        // terminal zero — so it goes to the sampler verbatim rather than being rebuilt through a lookalike scheduler,
        // which would differ by an ulp for general (i, N) and shift every default generation.
        ISampler sampler = FlowMatchSampling.Resolve(request.Scheduler, tsteps, seed, "HunyuanVideo");

        // Materialize the conditioning on the host so the per-step FreeActivations() below can't evict it (it is
        // re-uploaded fresh each step). promptEmbeds is already host (CropSequence), but pooled may be a live
        // GPU activation from the CLIP encoder — touching DataPointer forces the D2H sync + cache eviction.
        _ = promptEmbeds.DataPointer;
        _ = pooled.DataPointer;

        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();

            // Embedded/distilled guidance: ONE forward per step, no cond/uncond pair. The prediction therefore goes
            // back as the guidance-free aliased form at exactly 1.0 — substituting the embedded scale would not cancel
            // bit-exactly in F32 even though cond == uncond makes it mathematically irrelevant.
            DelegateDenoisePredictor predictor = new DelegateDenoisePredictor(
                PredictionType.FlowVelocity,
                (x, s, _) =>
                {
                    // The DiT conditions on the 0-1000 scale. The sigma array the sampler integrates over IS `tsteps`,
                    // so on-schedule this reproduces the old `tsteps[k] * 1000f` bit-for-bit, and a second-order
                    // sampler's intermediate sigma maps through the same expression instead of indexing a table.
                    Tensor v = _dit.Forward(Backend, x, promptEmbeds, pooled, s * 1000f, guidance);
                    return new DenoisePrediction(v, v);
                });
            if (k == 0)
            {
                sampler.Reset(latent.Shape);
            }
            // Plain flow-match Euler (z += v·(sigma_next − sigma)) is what EulerSampler collapses to, so a request
            // naming no sampler runs the same update the host `EulerCfgStep(latent, v, v, 1f, t − tsteps[k+1])` did.
            sampler.Step(Backend, latent, predictor, k);

            // The step now lands on the DEVICE (CfgEulerStep is in-place and leaves the latent GPU-resident), while
            // the FreeActivations sweep below reclaims device buffers WITHOUT a D2H sync-back — and the post-loop
            // DecodeLatent reads the latent on the host. Materialize it here so neither can lose the step. Costs
            // nothing net: the pre-sampler loop kept the latent host-side and D2H'd the velocity instead.
            _ = latent.DataPointer;

            sw.Stop();
            if (onProgress is not null)
            {
                onProgress.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
                {
                    Latent = latent,
                    LatentArch = LatentArchitecture.HunyuanVideo,
                });
            }
            // Reclaim GPU-resident activation buffers between steps: the DiT keeps intermediates on-device and any
            // not read-back/disposed linger until GC, accumulating to OOM (they held ~18 GB → the VAE decode OOM'd).
            // Safe: the sampler's device step is materialized to the host immediately above, so nothing cross-step is
            // GPU-only when this sweep frees the pool without syncing anything back.
            // trimPool:false — steps are identical, so the pool reservation is re-used verbatim; trimming here
            // released + re-mapped multiple GB of driver memory EVERY step.
            // SKIP on the step-graph path (HARTSY_DIT_GRAPH): the captured graph balances its own allocations and
            // holds the fixed input/output buffers at the addresses the capture bakes — freeing them here would
            // corrupt the replay. Re-checked each step so a mid-gen self-disable re-enables the sweep.
            if (!_dit.StepGraphActive) Backend.FreeActivations(trimPool: false);
        }

        Backend.Sync();
        // The captured step graph bakes the DiT weight device pointers; the frees below (and the next gen's
        // re-upload) would leave it replaying against freed memory — invalidate it before releasing the weights.
        _dit.InvalidateStepGraph(Backend);
        if (streamer is not null)
        {
            _dit.BeforeBlockForward = null;
            streamer.EvictAll();
            streamer.Dispose();
            Backend.FreeWeights(_dit.EnumerateSharedWeights());
            // Also purge the block weights: the streaming cache registers each uploaded block in the backend weight
            // cache, and any block still cached at the end (or not caught by EvictAll's state machine) would stay
            // resident and starve the VAE decode (was the decode OOM — ~23 GB DiT still held, <1 GB free).
            Backend.FreeWeights(_dit.EnumerateWeights());
        }
        else Backend.FreeWeights(_dit.EnumerateWeights());
        Backend.FreeActivations();
        Logs.Info($"HunyuanVideo: DiT freed, {Backend.FreeMemoryBytes() / (1024 * 1024)} MB free before VAE decode.");

        return latent;
    }

}
