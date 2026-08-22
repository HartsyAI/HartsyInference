using HartsyInference.Diffusion.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae.Mage;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Microsoft Mage-Flow text-to-image pipeline (NR-MMDiT, arXiv 2607.19064). Encodes the prompt through Qwen3-VL-4B (<see cref="LlamaStyleEncoder"/> final-layer <c>last_hidden_state</c>), packs the 128-channel latent into patch-1 tokens (identity — Mage-Flow does no patchify), runs the reused <see cref="QwenImageTransformer"/> (Mage-Flow config: 12 dual-stream blocks, image-only RoPE) under flow-match Euler with STATIC shift 6.0, and decodes through the bespoke one-step <see cref="MageVaeDecoder"/>.
/// <para>This is the lean correctness-first path: no block-streaming / step-cache / prompt-cache (those are perf
/// layers to add once GPU parity is confirmed). CFG is a dual forward when <c>cfgScale &gt; 1</c> (Turbo uses
/// cfgScale 1.0 → single forward). Build-blind: the seeded noise uses a host Gaussian (System.Random Box-Muller),
/// which will NOT bit-match torch's RNG — a reproducibility parity trap to revisit, not a correctness bug.</para></summary>
public sealed unsafe class MageFlowPipeline : DiffusionPipelineBase
{
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly QwenImageTransformer _transformer;
    private readonly MageVaeDecoder _vaeDecoder;
    private readonly MageVaeEncoder? _vaeEncoder;   // edit only (Mage-Flow-Edit-Turbo)
    private readonly QwenImageConfig _config;

    private const int VaeScale = 16;      // MageVAE downsample factor
    private const int LatentChannels = 128;
    private const float SchedulerShift = 6.0f;

    public MageFlowPipeline(IBackend backend, LlamaStyleEncoder textEncoder, QwenImageTransformer transformer,
        MageVaeDecoder vaeDecoder, QwenImageConfig config, MageVaeEncoder? vaeEncoder = null)
        : base(backend)
    {
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
    }

    /// <summary>Generates an image from already-tokenized (chat-templated) prompts. <paramref name="condDrop"/>/ <paramref name="uncondDrop"/> is the number of leading (system-prefix) tokens to discard from the encoder hidden states before they enter the DiT text stream (mirrors Krea2/Qwen-Image). Pass <paramref name="uncondTokens"/>=null (or cfgScale ≤ 1) for the guidance-free / Turbo path. Returns the decoded image as <c>[1, 3, H, W]</c> F32 in <c>[-1, 1]</c>.</summary>
    public Tensor GenerateFromTokens(int[] condTokens, int condDrop, int[]? uncondTokens, int uncondDrop,
        int width, int height, int steps, float cfgScale, long seed, Tensor? editRefPixels = null,
        string? seamlessTiling = null, long variationSeed = -1, double variationSeedStrength = 0,
        string? samplerSelection = null)
    {
        ThrowIfDisposed();
        // Wrap-pad every conv backend for this call so the output tiles seamlessly; restores on dispose. Passed
        // explicitly rather than read off a request — this pipeline takes primitives, not a TextToImageRequest.
        using IDisposable seamlessScope = BeginSeamlessTiling(seamlessTiling);
        bool useCfg = cfgScale > 1f && uncondTokens is not null;

        // 1. Text conditioning: Qwen3-VL-4B last_hidden_state, system prefix dropped.
        Tensor condHidden = EncodeDropped(condTokens, condDrop);
        Tensor? uncondHidden = useCfg ? EncodeDropped(uncondTokens!, uncondDrop) : null;

        // 1b. Edit: VAE-encode the reference image → packed ref tokens, appended in-context each forward. The DiT's
        // refGrids machinery gives them frame-axis-1 RoPE and drops them from the returned velocity. (The Qwen3-VL
        // vision-tower conditioning half of Mage-Flow-Edit is a follow-up; this is the in-context-latent half.)
        Tensor? refTokens = null;
        (int H, int W)[] refGrids = [];
        if (editRefPixels is not null && _vaeEncoder is not null)
        {
            Tensor refLatent = _vaeEncoder.Encode(Backend, editRefPixels);   // [1,128,rh,rw]
            int rh = (int)refLatent.Shape[2], rw = (int)refLatent.Shape[3];
            refTokens = PackPatch1(refLatent, LatentChannels);
            refLatent.Dispose();
            refGrids = [(rh, rw)];
            Backend.PreloadWeights(new List<Tensor> { refTokens });
        }

        // 2. Noisy latent [1, 128, h, w]; patch-1 packing → [1, h*w, 128].
        int h = height / VaeScale, w = width / VaeScale;
        Tensor latent = GaussianLatent(1, LatentChannels, h, w, seed);
        if (variationSeedStrength > 0)
        {
            // Same slerp blend every request-driven pipeline gets from TakeOrCreateNoise; passed explicitly here
            // because this pipeline takes primitives. The variation noise comes from SeedGenerator rather than
            // GaussianLatent's own RNG — both are unit-Gaussian Box-Muller, so the blend stays unit-variance.
            VariationNoise.BlendInPlace(latent, latent.Shape, variationSeed, variationSeedStrength);
        }
        Tensor packed = PackPatch1(latent, LatentChannels);
        latent.Dispose();

        // 3. Flow-match Euler with STATIC shift 6.0 (Z-Image schedule; not dynamic).
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(SchedulerShift);
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        // Sampler selection (2026-08-20). Mage-Flow takes primitives rather than a TextToImageRequest, so the
        // selection arrives as its own parameter threaded from MageFlowRecipePipeline. No step graph and no step
        // cache on this pipeline, so there is nothing to narrow.
        ISampler sampler = FlowMatchSampling.Resolve(samplerSelection, scheduler, unchecked((int)seed), "Mage-Flow");
        float[] timestepTable = timesteps.ToArray();
        Backend.PreloadWeights(_transformer.EnumerateWeights());

        Logs.Info($"[MageFlow] Denoise {steps} steps, CFG {cfgScale}, {width}x{height} (latent {w}x{h})" +
            (refTokens is not null ? " [edit]" : "") + ".");
        int patchDim = LatentChannels;
        DelegateDenoisePredictor predictor = new DelegateDenoisePredictor(
            PredictionType.FlowVelocity,
            (x, s, stepIndex) =>
            {
                // On-schedule sigmas reuse the loop's own `timesteps[i]/1000` expression — the F32 round trip
                // through x1000 is not exact, so raw sigma would shift every existing generation by an ulp.
                float normalizedT = stepIndex < steps && s == scheduler.SigmaAt(stepIndex)
                    ? timestepTable[stepIndex] / 1000f
                    : s;
                // Ref tokens are appended for this forward only (the DiT slices them off the velocity), and must
                // be rebuilt against whatever latent is being evaluated — a second-order sampler's sub-step latent
                // is not the loop's tensor.
                Tensor input = x;
                bool ownsInput = false;
                if (refTokens is not null)
                {
                    input = new(new TensorShape(1, x.Shape[1] + refTokens.Shape[1], patchDim), DType.F32);
                    Backend.Concat(input, new[] { x, refTokens }, 1);
                    ownsInput = true;
                }
                try
                {
                    Tensor cond = _transformer.Forward(Backend, input, condHidden, normalizedT, h, w, refGrids, refTimestepZero: true);
                    if (!useCfg)
                    {
                        return new DenoisePrediction(cond, cond);
                    }
                    Tensor uncond = _transformer.Forward(Backend, input, uncondHidden!, normalizedT, h, w, refGrids, refTimestepZero: true);
                    return new DenoisePrediction(cond, uncond, cfgScale);
                }
                finally
                {
                    if (ownsInput) input.Dispose();
                }
            });
        sampler.Reset(packed.Shape);
        for (int i = 0; i < steps; i++)
        {
            sampler.Step(Backend, packed, predictor, i);
        }
        condHidden.Dispose();
        uncondHidden?.Dispose();
        refTokens?.Dispose();

        // 4. Unpack → [1, 128, h, w], decode → [1, 3, H, W] in [-1, 1].
        Tensor finalLatent = UnpackPatch1(packed, LatentChannels, h, w);
        packed.Dispose();
        Backend.FreeWeights(_transformer.EnumerateWeights());
        Tensor image = _vaeDecoder.Decode(Backend, finalLatent);
        finalLatent.Dispose();
        return image;
    }

    // Encode tokens through Qwen3-VL-4B; drop the leading system-prefix rows from the [1, S, 2560] hidden states.
    private Tensor EncodeDropped(int[] tokens, int drop)
    {
        Tensor full = _textEncoder.Encode(Backend, new[] { tokens });   // [1, S, 2560]
        Backend.Sync();
        if (drop <= 0) return full;
        int s = (int)full.Shape[1], d = (int)full.Shape[2];
        int kept = s - drop;
        Tensor sliced = new(new TensorShape(1, kept, d), DType.F32);
        Buffer.MemoryCopy((byte*)full.DataPointer + (long)drop * d * 4, (void*)sliced.DataPointer,
            (long)kept * d * 4, (long)kept * d * 4);
        full.Dispose();
        return sliced;
    }

    // patch_size=1: [1,C,h,w] → [1, h*w, C] (token = one latent cell, channels-last).
    private static Tensor PackPatch1(Tensor latent, int c)
    {
        int h = (int)latent.Shape[2], w = (int)latent.Shape[3], hw = h * w;
        Tensor packed = new(new TensorShape(1, hw, c), DType.F32);
        float* src = (float*)latent.DataPointer; float* dst = (float*)packed.DataPointer;
        for (int p = 0; p < hw; p++)
            for (int ch = 0; ch < c; ch++)
                dst[(long)p * c + ch] = src[(long)ch * hw + p];
        return packed;
    }

    private static Tensor UnpackPatch1(Tensor packed, int c, int h, int w)
    {
        int hw = h * w;
        Tensor latent = new(new TensorShape(1, c, h, w), DType.F32);
        float* src = (float*)packed.DataPointer; float* dst = (float*)latent.DataPointer;
        for (int p = 0; p < hw; p++)
            for (int ch = 0; ch < c; ch++)
                dst[(long)ch * hw + p] = src[(long)p * c + ch];
        return latent;
    }

    private static Tensor GaussianLatent(int b, int c, int h, int w, long seed)
    {
        Tensor t = new(new TensorShape(b, c, h, w), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(unchecked((int)seed));
        long n = (long)b * c * h * w;
        for (long i = 0; i < n; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = 1.0 - rng.NextDouble();
            p[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return t;
    }

    protected override void DisposeCore() { }
}
