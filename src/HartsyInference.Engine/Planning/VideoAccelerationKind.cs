namespace HartsyInference.Engine.Planning;

/// <summary>The checkpoint-bound acceleration recipe selected during video planning.</summary>
public enum VideoAccelerationKind
{
    /// <summary>The base denoising recipe.</summary>
    None = 0,

    /// <summary>A distilled Turbo checkpoint or adapter with locked sampling settings.</summary>
    Turbo = 1,

    /// <summary>Piecewise distillation with sigma-dependent output-head banks.</summary>
    Pdd = 2,

    /// <summary>Four-evaluation FastH3 execution with learned video sparse attention.</summary>
    Vsa = 3,
}
