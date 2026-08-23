namespace HartsyInference.Diffusion.Schedulers;

/// <summary>Shared configuration for all diffusion schedulers.</summary>
public sealed record SchedulerConfig
{
    /// <summary>Number of training timesteps the model was trained with. Default: 1000.</summary>
    public int NumTrainTimesteps { get; init; } = 1000;

    /// <summary>Starting beta value for the noise schedule.</summary>
    public float BetaStart { get; init; } = 0.00085f;

    /// <summary>Ending beta value for the noise schedule.</summary>
    public float BetaEnd { get; init; } = 0.012f;

    /// <summary>Type of beta schedule. "scaled_linear" is default for Stable Diffusion.</summary>
    public BetaScheduleType BetaSchedule { get; init; } = BetaScheduleType.ScaledLinear;

    /// <summary>Model prediction type: "epsilon" predicts noise, "v_prediction" predicts velocity.</summary>
    public PredictionType PredictionType { get; init; } = PredictionType.Epsilon;

    /// <summary>Which of the three <see cref="Schedulers.TimestepSpacing"/> stride formulas selects the training-schedule subset used at inference.</summary>
    public TimestepSpacing TimestepSpacing { get; init; } = TimestepSpacing.Leading;
}

/// <summary>Beta noise schedule types.</summary>
public enum BetaScheduleType : byte
{
    /// <summary>Linear interpolation of beta values.</summary>
    Linear = 0,

    /// <summary>Linear interpolation of sqrt(beta) values, then squared. Default for Stable Diffusion.</summary>
    ScaledLinear = 1,
}

/// <summary>Model prediction output type — what a denoiser's raw output <em>means</em>, which every sampler beyond
/// plain Euler has to convert to a denoised (x0) estimate before its own math applies.</summary>
public enum PredictionType : byte
{
    /// <summary>Model predicts the noise (epsilon). Used by SD1.5, SDXL.</summary>
    Epsilon = 0,

    /// <summary>Model predicts the velocity (v). Used by some fine-tuned models.</summary>
    VPrediction = 1,

    /// <summary>Model predicts the flow-matching velocity <c>v = noise − x0</c> on a rectified path, with sigma running
    /// over [0,1] rather than the VP noise scale. Flux, SD3, Qwen-Image, Krea2, Z-Image and the rest of the DiT fleet.
    /// <para>Shares Epsilon's denoised formula (<c>x0 = x − sigma·pred</c>) — the reason one Euler step serves both, and
    /// why <see cref="Core.Backends.IBackend.CfgEulerStep"/> is the single seam every family already funnels through.
    /// It is a distinct member anyway because the sigma domain differs, so anything reading sigma as a VP noise level
    /// (the alphas-cumprod round trip, <c>ScaleModelInput</c>) must not treat the two alike.</para></summary>
    FlowVelocity = 2,

    /// <summary>Flow-matching velocity with the sign FLIPPED: the model predicts <c>−v</c>, so the Euler update is
    /// <c>x += pred·(−dt)</c> and the denoised estimate is <c>x0 = x + sigma·pred</c>.
    /// <para>This is the NextDiT convention (Z-Image, Lumina-Image 2.0), which folds diffusers' mandatory
    /// <c>noise_pred = −noise_pred</c> into the step delta instead of negating the tensor — visible in those pipelines
    /// as <c>CfgEulerStep(..., −scheduler.Dt(i))</c>. Modelling it as its own domain rather than negating the
    /// prediction tensor keeps the default path free of an extra kernel and allocation per forward: flipping a scalar
    /// coefficient is exact in F32 and costs nothing, whereas <c>Scale(−1)</c> on a latent-sized tensor is real work on
    /// every step of every generation.</para></summary>
    NegatedFlowVelocity = 3,
}

/// <summary>Strategy for selecting timesteps during inference.</summary>
public enum TimestepSpacing : byte
{
    /// <summary>Evenly spaced from 0, stepping by train_timesteps / inference_steps.</summary>
    Leading = 0,

    /// <summary>Evenly spaced from train_timesteps down to 0, stepping by train_timesteps / inference_steps.</summary>
    Trailing = 1,

    /// <summary>Linearly interpolated between 0 and train_timesteps-1.</summary>
    Linspace = 2,
}
