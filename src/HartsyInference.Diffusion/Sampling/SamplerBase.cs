using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>Shared plumbing for samplers ported from ComfyUI: a local sigma view starting at the first step the run
/// actually takes, the flow first-sigma offset, noise-source ownership and pinned history.</summary>
/// <remarks>ComfyUI truncates the sigma array for img2img, so its samplers index from zero at the first step taken.
/// Subclasses see the same view through <see cref="Sigma"/> and the local index passed to <see cref="StepLocal"/>.</remarks>
public abstract class SamplerBase : ISampler
{
    private readonly float[] _sigmas;
    private float[] _local = [];
    private int _firstStep = -1;
    private bool _ready;

    /// <summary>Validates and stores the resolved schedule.</summary>
    protected SamplerBase(float[] sigmas, int seed, SamplerOptions? options)
    {
        ArgumentNullException.ThrowIfNull(sigmas);
        if (sigmas.Length < 2)
        {
            throw new ArgumentException($"Need at least 2 sigmas (one step plus the terminal zero); got {sigmas.Length}.",
                nameof(sigmas));
        }
        foreach (float s in sigmas)
        {
            if (!float.IsFinite(s) || s < 0f)
            {
                throw new ArgumentException($"Sigmas must be finite and non-negative; got {s}.", nameof(sigmas));
            }
        }
        _sigmas = sigmas;
        Seed = seed;
        Options = options ?? SamplerOptions.Default;
    }

    /// <inheritdoc/>
    public abstract string Name { get; }

    /// <inheritdoc/>
    public int StepCount => _sigmas.Length - 1;

    /// <summary>Base seed for stochastic draws.</summary>
    protected int Seed { get; }

    /// <summary>Family mapping and test seams.</summary>
    protected SamplerOptions Options { get; }

    /// <summary>Latent shape from the last <see cref="Reset"/>.</summary>
    protected TensorShape Shape { get; private set; }

    /// <summary>Whether the model is a flow (ComfyUI <c>CONST</c>) model; known from the first step.</summary>
    protected bool IsFlow { get; private set; }

    /// <summary>Noise for this run, or null when <see cref="Noise"/> is <see cref="NoiseKind.None"/>.</summary>
    protected INoiseSource? NoiseSource { get; private set; }

    /// <summary>Number of entries in the local sigma view, including the terminal one.</summary>
    protected int LocalLength => _local.Length;

    /// <summary>Whether ComfyUI applies <c>offset_first_sigma_for_snr</c> for this sampler.</summary>
    protected virtual bool OffsetsFirstSigma => false;

    /// <summary>Which noise source the sampler draws from.</summary>
    protected virtual NoiseKind Noise => NoiseKind.None;

    /// <inheritdoc/>
    public void Reset(TensorShape latentShape)
    {
        Shape = latentShape;
        _ready = true;
        _firstStep = -1;
        _local = [];
        NoiseSource?.Dispose();
        NoiseSource = null;
        OnReset();
    }

    /// <inheritdoc/>
    public void Step(IBackend backend, Tensor z, IDenoisePredictor predictor, int stepIndex)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(z);
        ArgumentNullException.ThrowIfNull(predictor);
        if (!_ready)
        {
            throw new InvalidOperationException($"{Name}: call Reset with the latent shape before the first Step.");
        }
        if (_firstStep < 0)
        {
            Begin(predictor, stepIndex);
        }
        int local = stepIndex - _firstStep;
        if (local < 0 || local >= _local.Length - 1)
        {
            throw new ArgumentOutOfRangeException(nameof(stepIndex),
                $"{Name}: step {stepIndex} is outside the run that started at step {_firstStep}; call Reset first.");
        }
        StepLocal(backend, z, predictor, local, stepIndex);
    }

    /// <summary>Advances <paramref name="z"/> from local sigma <paramref name="i"/> to <c>i + 1</c>.</summary>
    /// <param name="stepIndex">The pipeline's schedule index, for model evaluations and noise keys.</param>
    protected abstract void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex);

    /// <summary>Clears per-run state; called from <see cref="Reset"/>.</summary>
    protected virtual void OnReset()
    {
    }

    /// <summary>Precomputes per-run tables once the local view and model type are known.</summary>
    protected virtual void OnBegin()
    {
    }

    /// <summary>Local sigma <paramref name="i"/>, ComfyUI's <c>sigmas[i]</c>.</summary>
    protected float Sigma(int i) => _local[i];

    /// <summary>The local sigma view; read-only by contract.</summary>
    protected float[] LocalSigmas => _local;

    /// <summary>ComfyUI's <c>model_sampling.percent_to_sigma</c> for this family.</summary>
    protected double SigmaAtPercent(double percent) => Options.SigmaAtPercent(percent, _sigmas, IsFlow);

    /// <summary>Half-log-SNR of <paramref name="sigma"/> for this model type.</summary>
    protected double Lambda(double sigma) => SamplerMath.HalfLogSnr(sigma, IsFlow);

    /// <summary>Sigma of half-log-SNR <paramref name="lambda"/> for this model type.</summary>
    protected double SigmaOfLambda(double lambda) => SamplerMath.HalfLogSnrToSigma(lambda, IsFlow);

    /// <summary>Evaluates the model's denoised estimate at <paramref name="sigma"/>; the caller owns the result.</summary>
    protected static Tensor Denoise(IBackend backend, IDenoisePredictor predictor, Tensor x, float sigma, int stepIndex) =>
        SamplerMath.PredictDenoised(backend, predictor, x, sigma, stepIndex);

    /// <summary>In-place <c>target += scale·N(0,1)</c> for the interval <c>sigma → sigmaNext</c>.</summary>
    protected void AddNoise(IBackend backend, Tensor target, int stepIndex, int subDraw, float sigma, float sigmaNext, float scale)
    {
        if (NoiseSource is null)
        {
            throw new InvalidOperationException($"{Name} declared no noise source but tried to draw noise.");
        }
        SamplerOps.AddNoise(backend, target, NoiseSource, stepIndex, subDraw, sigma, sigmaNext, scale);
    }

    /// <summary>Draws a noise tensor for the interval; the caller owns it.</summary>
    protected Tensor DrawNoise(int stepIndex, int subDraw, float sigma, float sigmaNext)
    {
        if (NoiseSource is null)
        {
            throw new InvalidOperationException($"{Name} declared no noise source but tried to draw noise.");
        }
        return NoiseSource.Sample(stepIndex, subDraw, sigma, sigmaNext);
    }

    /// <summary>Replaces <paramref name="slot"/> with <paramref name="value"/>, pinned so it survives the pipeline's
    /// mid-loop activation release.</summary>
    protected static void Keep(IBackend backend, ref Tensor? slot, Tensor value)
    {
        ArgumentNullException.ThrowIfNull(backend);
        slot?.Dispose();
        slot = value;
        backend.PinActivation(value);
    }

    /// <summary>Disposes and clears a history slot.</summary>
    protected static void Release(ref Tensor? slot)
    {
        slot?.Dispose();
        slot = null;
    }

    private void Begin(IDenoisePredictor predictor, int stepIndex)
    {
        if (stepIndex < 0 || stepIndex >= _sigmas.Length - 1)
        {
            throw new ArgumentOutOfRangeException(nameof(stepIndex), $"{Name}: step {stepIndex} is outside the schedule.");
        }
        _firstStep = stepIndex;
        IsFlow = SamplerMath.IsFlow(predictor.Prediction);
        _local = _sigmas[stepIndex..];
        float[] unshifted = _local;
        if (OffsetsFirstSigma && IsFlow && _local.Length > 1 && _local[0] >= 1f)
        {
            _local = (float[])_local.Clone();
            _local[0] = (float)SigmaAtPercent(1e-4);
        }
        NoiseSource = Noise switch
        {
            NoiseKind.None => null,
            _ when Options.NoiseFactory is not null => Options.NoiseFactory(Shape, unshifted),
            NoiseKind.Gaussian => new SeededNoiseSource(Shape, Seed),
            NoiseKind.Brownian => BrownianNoiseSource.ForSchedule(Shape, Seed, unshifted),
            _ => throw new NotSupportedException($"Unhandled noise kind {Noise}."),
        };
        OnBegin();
    }

    /// <summary>Noise a sampler draws.</summary>
    protected enum NoiseKind
    {
        /// <summary>Deterministic sampler.</summary>
        None,

        /// <summary>Independent seeded Gaussian per draw.</summary>
        Gaussian,

        /// <summary>Correlated Brownian increments.</summary>
        Brownian,
    }
}
