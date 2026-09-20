using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Qwen-Image 2.1 text-to-image pipeline. Encodes the prompt with Qwen3-VL-8B (last decoder layer, no final
/// norm), runs the single-stream <see cref="QwenImage21Transformer"/> under flow-match Euler, and decodes through the
/// Wan 2.2 VAE parameterized for this model — 64-channel latent, 16× spatial, patch 1, temporal kernel 1, and a
/// <b>four-channel</b> output, because Qwen-Image 2.1 generates transparency natively rather than matting it
/// afterwards.
///
/// <para>The schedule is <c>ModelSamplingFlux</c> at ComfyUI's fixed <c>shift = 0.69</c>. That shift is an
/// <c>exp(mu)</c> exponent, and <c>flux_time_shift(mu, 1, t)</c> is algebraically
/// <c>e^mu·t / (1 + (e^mu − 1)·t)</c> — the same curve <see cref="FlowMatchEulerDiscreteScheduler"/> already
/// implements, so it is reused with <c>shift = e^0.69</c> rather than duplicated. The conditioning timestep IS the
/// sigma (<c>ModelSamplingFlux.timestep</c> is the identity); the DiT applies the ×1000 itself.</para>
///
/// <para>The text prefix is encoded through the DiT once per prompt into a
/// <see cref="QwenImage21PrefixCache"/> and reused for every step — see
/// <see cref="QwenImage21Transformer"/> for why that is exact rather than an approximation. Under CFG each branch
/// gets its own cache.</para></summary>
public sealed unsafe class QwenImage21Pipeline : DiffusionPipelineBase
{
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly QwenImage21Transformer _transformer;
    private readonly Wan22VaeDecoder _vaeDecoder;
    private readonly QwenImage21Config _config;

    /// <summary>VAE spatial reduction; the DiT does no patchify, so this is also the token grid's stride.</summary>
    public const int VaeScale = 16;

    /// <summary>ComfyUI <c>QwenImage21.sampling_settings["shift"]</c>, its mu at 1024×1024.</summary>
    public const float SchedulerMu = 0.69f;

    public QwenImage21Pipeline(IBackend backend, LlamaStyleEncoder textEncoder, QwenImage21Transformer transformer,
        Wan22VaeDecoder vaeDecoder, QwenImage21Config config)
        : base(backend)
    {
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates from already-templated token ids. <paramref name="condDrop"/> is the number of leading
    /// template tokens to discard from the encoder's hidden states — Qwen-Image 2.1 drops everything before the
    /// SECOND <c>&lt;|im_start|&gt;</c>, i.e. the whole system turn. Returns the decoded image as
    /// <c>[1, 4, H, W]</c> F32 in <c>[-1, 1]</c>: channel 3 is the model's own alpha.</summary>
    public Tensor GenerateFromTokens(int[] condTokens, int condDrop, int[]? uncondTokens, int uncondDrop,
        int width, int height, int steps, float cfgScale, long seed, string? seamlessTiling = null,
        long variationSeed = -1, double variationSeedStrength = 0, string? samplerSelection = null,
        Action<GenerationProgress>? onProgress = null,
        Prompting.WeightedTokenSequence? condWeights = null, Prompting.WeightedTokenSequence? uncondWeights = null)
    {
        ThrowIfDisposed();
        RequireMatchingWeights(condTokens, condWeights, nameof(condWeights));
        using IDisposable seamlessScope = BeginSeamlessTiling(seamlessTiling);
        bool useCfg = cfgScale > 1f && uncondTokens is not null;
        if (useCfg)
        {
            RequireMatchingWeights(uncondTokens!, uncondWeights, nameof(uncondWeights));
        }

        int h = height / VaeScale, w = width / VaeScale;

        // 1. Text conditioning. The encoder and the DiT are ~17 GB and ~14 GB at bf16 and do not overlap in time,
        // so the encoder is evicted before the DiT is staged — both resident at once does not fit a 24 GB card.
        // The projected text rows and their per-block K/V are step-independent, so this is the only time the text
        // touches the transformer.
        QwenImage21PrefixCache condPrefix;
        QwenImage21PrefixCache? uncondPrefix = null;
        Tensor? condHidden = null, uncondHidden = null;
        try
        {
            condHidden = ApplyTokenWeights(EncodeDropped(condTokens, condDrop), condWeights);
            if (useCfg)
            {
                uncondHidden = ApplyTokenWeights(EncodeDropped(uncondTokens!, uncondDrop), uncondWeights);
            }
            Backend.FreeWeights(_textEncoder.EnumerateWeights());
            Backend.PreloadWeights(_transformer.EnumerateWeights());
            condPrefix = _transformer.BuildPrefix(Backend, condHidden);
            if (uncondHidden is not null)
            {
                try
                {
                    uncondPrefix = _transformer.BuildPrefix(Backend, uncondHidden);
                }
                catch
                {
                    condPrefix.Dispose();
                    throw;
                }
            }
        }
        finally
        {
            condHidden?.Dispose();
            uncondHidden?.Dispose();
        }

        // From here to the decode, everything runs under one finally that releases the prefix caches. They are
        // depth x 2 x T x hidden of resident GPU memory each, and the setup below can throw on caller-supplied
        // input — FlowMatchSampling.Resolve refuses an unrecognized sampler name — so a guard that started only
        // at the denoise loop would leak both caches on every misspelled sampler.
        Tensor? latent = null;
        try
        {
            latent = GaussianLatent(1, _config.InChannels, h, w, seed);
            if (variationSeedStrength > 0)
            {
                VariationNoise.BlendInPlace(latent, latent.Shape, variationSeed, variationSeedStrength);
            }

            FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(MathF.Exp(SchedulerMu));
            scheduler.SetTimesteps(steps);
            ISampler sampler = FlowMatchSampling.Resolve(samplerSelection, scheduler, unchecked((int)seed), "Qwen-Image 2.1");

            Logs.Info($"[QwenImage21] Denoise {steps} steps, CFG {cfgScale}, {width}x{height} (latent {w}x{h}, "
                + $"{h * w} image tokens, {condPrefix.Length} text tokens).");

            DelegateDenoisePredictor predictor = new DelegateDenoisePredictor(
                PredictionType.FlowVelocity,
                (x, s, stepIndex) =>
                {
                    Tensor cond = _transformer.Forward(Backend, x, condPrefix, s);
                    if (!useCfg)
                    {
                        return new DenoisePrediction(cond, cond);
                    }
                    Tensor uncond = _transformer.Forward(Backend, x, uncondPrefix!, s);
                    return new DenoisePrediction(cond, uncond, cfgScale);
                });

            sampler.Reset(latent.Shape);
            for (int i = 0; i < steps; i++)
            {
                sampler.Step(Backend, latent, predictor, i);
                onProgress?.Invoke(new GenerationProgress(i + 1, steps, 0)
                {
                    Latent = latent,
                    LatentArch = LatentArchitecture.QwenImage21,
                });
            }

            Backend.FreeWeights(_transformer.EnumerateWeights());

            // 2. Decode. The Wan 2.2 decoder is a video decoder, so the latent carries a length-1 time axis; T=1 is
            // its stateless first-chunk path. The latent denorm (this model's own 64-channel table) is inside.
            using Tensor latent5d = latent.Reshape(new TensorShape([1L, _config.InChannels, 1L, h, w]));
            using Tensor decoded = _vaeDecoder.Decode(Backend, latent5d);

            // Drop the length-1 time axis by copying rather than viewing: a reshaped view aliases the parent's
            // buffer, and the caller would have no handle on the parent to release. One frame is cheap.
            int outChannels = (int)decoded.Shape[1];
            Tensor image = new Tensor(new TensorShape(1, outChannels, height, width), decoded.DType);
            Backend.SliceRowsGeneric(image, decoded, rowOffset: 0);
            return image;
        }
        finally
        {
            condPrefix.Dispose();
            uncondPrefix?.Dispose();
            latent?.Dispose();
        }
    }

    /// <summary>Encodes the prompt and drops the leading template rows. The tap is the last decoder layer
    /// <b>without</b> the final norm (ComfyUI <c>layer_norm_hidden_state = False</c>), which the encoder config
    /// expresses as <c>HasFinalNorm = false</c>.</summary>
    private Tensor EncodeDropped(int[] tokens, int drop)
    {
        Tensor hidden = _textEncoder.Encode(Backend, [tokens]);
        if (drop <= 0)
        {
            return hidden;
        }
        int seq = (int)hidden.Shape[hidden.Shape.Rank - 2];
        int dim = (int)hidden.Shape[hidden.Shape.Rank - 1];
        if (drop >= seq)
        {
            hidden.Dispose();
            throw new ArgumentOutOfRangeException(nameof(drop), drop,
                $"The template prefix is {drop} tokens but the encoded prompt is only {seq}; nothing would condition the image.");
        }
        Tensor kept = new Tensor(new TensorShape(1, seq - drop, dim), hidden.DType);
        Backend.SliceRowsGeneric(kept, hidden, drop);
        hidden.Dispose();
        return kept;
    }

    /// <summary>Scales each token's cond row by its weight after the template drop — SwarmUI's CondScale mechanism,
    /// whose right-alignment offset is negative here because of that drop. Adopts and disposes
    /// <paramref name="hidden"/>.</summary>
    private Tensor ApplyTokenWeights(Tensor hidden, Prompting.WeightedTokenSequence? weights)
    {
        if (weights is null || weights.IsUniformlyUnweighted)
        {
            return hidden;
        }
        Tensor? scaled = Prompting.CondTokenWeights.Apply(Backend, hidden, null, weights).Cond;
        if (scaled is null)
        {
            return hidden;
        }
        hidden.Dispose();
        return scaled;
    }

    /// <summary>Weights are matched to conditioning rows by position, so an array that does not describe the tokens
    /// it arrived with would silently shift every emphasis onto a neighbouring word rather than fail.</summary>
    private static void RequireMatchingWeights(int[] tokenIds, Prompting.WeightedTokenSequence? weights, string name)
    {
        if (weights is not null && weights.Weights.Length != tokenIds.Length)
        {
            throw new ArgumentException(
                $"Weights describe {weights.Weights.Length} tokens but {tokenIds.Length} were passed.", name);
        }
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
