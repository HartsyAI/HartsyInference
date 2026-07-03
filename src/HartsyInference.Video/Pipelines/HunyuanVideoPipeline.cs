using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
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
public sealed unsafe class HunyuanVideoPipeline : DiffusionPipelineBase
{
    private const int VaeSpatial = 8;
    private const int VaeTemporal = 4;
    private const float VaeScalingFactor = 0.476986f;

    private readonly HunyuanVideoDit _dit;
    private readonly HunyuanVideoVaeDecoder _vae;
    private readonly HunyuanVideoConfig _config;

    public HunyuanVideoPipeline(IBackend backend, HunyuanVideoDit dit, HunyuanVideoVaeDecoder vae, HunyuanVideoConfig config)
        : base(backend)
    {
        _dit = dit;
        _vae = vae;
        _config = config;
    }

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
            streamer = new BlockStreamingController(Backend.StreamingCache!, blocks, prefetchAhead: 2, retainBehind: 0);
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

        // Materialize the conditioning on the host so the per-step FreeActivations() below can't evict it (it is
        // re-uploaded fresh each step). promptEmbeds is already host (CropSequence), but pooled may be a live
        // GPU activation from the CLIP encoder — touching DataPointer forces the D2H sync + cache eviction.
        _ = promptEmbeds.DataPointer;
        _ = pooled.DataPointer;

        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float t = tsteps[k], dt = t - tsteps[k + 1];
            float tEmb = t * 1000f;
            Tensor v = _dit.Forward(Backend, latent, promptEmbeds, pooled, tEmb, guidance);
            // Embedded-guidance path: single velocity, plain flow-match Euler (z -= v·dt).
            LancePipelineCommon.EulerCfgStep(latent, v, v, 1f, dt);
            v.Dispose();
            sw.Stop();
            if (onProgress is not null)
            {
                Tensor preview = ExtractMiddleFrame(latent, tLat, hLat, wLat, _config.OutChannels);
                onProgress.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
                {
                    Latent = preview,
                    LatentArch = LatentArchitecture.Ltx,
                });
                preview.Dispose();
            }
            // Reclaim GPU-resident activation buffers between steps: the DiT keeps intermediates on-device and any
            // not read-back/disposed linger until GC, accumulating to OOM (they held ~18 GB → the VAE decode OOM'd).
            // Safe: EulerCfgStep updates the latent in-place on the host, so nothing cross-step is GPU-only.
            // trimPool:false — steps are identical, so the pool reservation is re-used verbatim; trimming here
            // released + re-mapped multiple GB of driver memory EVERY step.
            Backend.FreeActivations(trimPool: false);
        }

        Backend.Sync();
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

    /// <summary>Middle latent frame <c>[1, C, H, W]</c> from a <c>[1, C, T, H, W]</c> latent, for latent2rgb previews.</summary>
    private static Tensor ExtractMiddleFrame(Tensor latent, int t, int h, int w, int channels)
    {
        Tensor outT = new(new TensorShape([1L, channels, h, w]), DType.F32);
        float* sp = (float*)latent.DataPointer; float* dp = (float*)outT.DataPointer;
        long spatial = (long)t * h * w;
        int mid = t / 2;
        for (int c = 0; c < channels; c++)
            for (int hi = 0; hi < h; hi++)
                for (int wi = 0; wi < w; wi++)
                    dp[((long)c * h + hi) * w + wi] = sp[(long)c * spatial + ((long)mid * h + hi) * w + wi];
        return outT;
    }
}
