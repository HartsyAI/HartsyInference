using HartsyInference.Core.Backends;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.Diffusion.Pipelines.MovieRoll;

/// <summary>MovieRoll for one prompt, from text features to frames: the rolling sampler over the latent grid, then the unscale by the VAE's
/// scaling factor and the VAE decode. The DiT and the VAE are shared resources the caller owns and loads; the pipeline owns only the
/// per-run KV cache it creates. Text features come in as the encoders' outputs, so this class has no tokenizer or encoder.
/// The sampler, the schedule and the decode are each covered by their own tests; this composition is checked on the tiny config.</summary>
public sealed class MovieRollPipeline : DiffusionPipelineBase
{
    private readonly HunyuanVideo15Dit _dit;
    private readonly HunyuanVideo15Vae _vae;
    private readonly MovieRollSchedule _schedule;
    private readonly long _kvByteBudget;

    /// <summary>Activation headroom the block streamer keeps free, as the HunyuanVideo pipeline's: the sampler's windows and the decode's tiles.</summary>
    private const long ActivationHeadroomBytes = 6L << 30;

    /// <param name="kvByteBudget">The most K and V bytes a run may hold in its cache; <see cref="long.MaxValue"/> means unbounded.
    /// A run that would exceed it is refused before any write.</param>
    public MovieRollPipeline(IBackend backend, HunyuanVideo15Dit dit, HunyuanVideo15Vae vae, MovieRollSchedule schedule, long kvByteBudget = long.MaxValue)
        : base(backend)
    {
        _dit = dit ?? throw new ArgumentNullException(nameof(dit));
        _vae = vae ?? throw new ArgumentNullException(nameof(vae));
        _schedule = schedule ?? throw new ArgumentNullException(nameof(schedule));
        if (kvByteBudget <= 0) throw new ArgumentOutOfRangeException(nameof(kvByteBudget), kvByteBudget, "The K and V budget is positive.");
        _kvByteBudget = kvByteBudget;
    }

    /// <summary>Runs the rolling sampler over the latent grid and returns the clean latents <c>[1, C, T, H, W]</c>, owned by the caller. <paramref name="noise"/>
    /// draws a latent <c>[1, C, frames, latentHeight, latentWidth]</c> for each chunk and each context-noise pass, in call order (see <see cref="MovieRollSampler.Sample"/>).</summary>
    public Tensor SampleLatents(MovieRollSamplerOptions options, Tensor qwen, Tensor? byt5, Tensor? negativeQwen, Tensor? negativeByt5,
        int latentHeight, int latentWidth, Func<int, Tensor> noise)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(noise);
        if (latentHeight <= 0) throw new ArgumentOutOfRangeException(nameof(latentHeight), "The latent height is positive.");
        if (latentWidth <= 0) throw new ArgumentOutOfRangeException(nameof(latentWidth), "The latent width is positive.");

        using MovieRollKvCache cache = new(Backend, _kvByteBudget);
        cache.Init(_dit.StreamLayerNames(), latentHeight, latentWidth);
        // The DiT's blocks stream through BlockStreamingScope on a backend with a streaming cache, as the HunyuanVideo pipeline does: the shared
        // weights and a resident prefix are preloaded, and the rest of the blocks are staged one at a time as the forward reaches them. A backend
        // without a streaming cache preloads the whole DiT here.
        BlockStreamingScope stream = BlockStreamingScope.Open(new BlockStreamingOptions
        {
            Backend = Backend,
            Denoiser = _dit,
            ModelName = "MovieRoll",
            HeadroomBytes = ActivationHeadroomBytes,
        });
        try
        {
            MovieRollSampler sampler = new(Backend, _dit, _schedule) { StepCompleted = stream.EndStep };
            return sampler.Sample(cache, options, qwen, byt5, negativeQwen, negativeByt5, noise);
        }
        finally
        {
            // Unhooks the streamer and evicts its window, then finishes the device work that reads the DiT before its weights are released.
            stream.Dispose();
            Backend.Sync();
            Backend.FreeWeights(_dit.EnumerateWeights());
        }
    }

    /// <summary>Samples, unscales and decodes one clip to frames <c>[1, 3, T', 16·H, 16·W]</c> owned by the caller. Same noise contract as <see cref="SampleLatents"/>.</summary>
    public Tensor Generate(MovieRollSamplerOptions options, Tensor qwen, Tensor? byt5, Tensor? negativeQwen, Tensor? negativeByt5,
        int latentHeight, int latentWidth, Func<int, Tensor> noise)
    {
        // The latents come from the DiT's backend, and Scale and Decode run on it too. A separate VAE backend would need a host hop between
        // the two, which this pipeline does not do yet, so it is refused rather than silently ignored.
        if (!ReferenceEquals(VaeBackend, Backend))
            throw new NotSupportedException("MovieRollPipeline decodes on the DiT backend only; a separate VaeBackend is not wired.");
        using Tensor latents = SampleLatents(options, qwen, byt5, negativeQwen, negativeByt5, latentHeight, latentWidth, noise);
        // The decoder takes the latent already divided by the scaling factor (movieroll_pipeline.py:decode_latents, latents / scaling_factor).
        using Tensor unscaled = new(latents.Shape, DType.F32);
        Backend.Scale(unscaled, latents, 1f / _vae.Config.ScalingFactor);
        // The VAE's weights are resident only for the decode, so the sampling run's memory is released before it.
        Backend.PreloadWeights(_vae.EnumerateWeights());
        try
        {
            return _vae.Decode(Backend, unscaled);
        }
        finally
        {
            Backend.Sync();
            Backend.FreeWeights(_vae.EnumerateWeights());
        }
    }

    /// <inheritdoc/>
    protected override void DisposeCore()
    {
        // The DiT, VAE and schedule are shared and stay alive; the pipeline holds no state between runs.
    }
}
