namespace SharpInference.Diffusion.Schedulers;

/// <summary>Shared configuration for all diffusion schedulers.</summary>
public sealed class SchedulerConfig
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

    /// <summary>Timestep spacing strategy for inference.</summary>
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

/// <summary>Model prediction output type.</summary>
public enum PredictionType : byte
{
    /// <summary>Model predicts the noise (epsilon). Used by SD1.5, SDXL.</summary>
    Epsilon = 0,

    /// <summary>Model predicts the velocity (v). Used by some fine-tuned models.</summary>
    VPrediction = 1,
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
