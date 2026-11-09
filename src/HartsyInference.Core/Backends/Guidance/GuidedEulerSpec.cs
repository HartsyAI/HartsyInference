namespace HartsyInference.Core.Backends.Guidance;

/// <summary>One guided Euler step for <see cref="IBackend.GuidedEulerStep"/>: the guided prediction is formed by
/// <see cref="Form"/> with <see cref="Guidance"/>, rescaled by <see cref="Norm"/>, and added to the latent times
/// <see cref="Delta"/>.</summary>
public readonly record struct GuidedEulerSpec(float Guidance, float Delta, GuidanceForm Form, GuidanceNorm Norm)
{
    /// <summary>The plain step: <c>z += (g · cond + (1 − g) · uncond) · delta</c>. With both predictions the same tensor and
    /// <paramref name="guidance"/> 1 it is an unguided Euler step.</summary>
    public static GuidedEulerSpec Lerp(float guidance, float delta) => new(guidance, delta, GuidanceForm.Lerp, GuidanceNorm.None);

    /// <summary>The globally renormalized step (Lance): <c>uncond + g · (cond − uncond)</c>, rescaled by one ratio over the whole
    /// tensor clamped to <c>[min, max]</c>.</summary>
    public static GuidedEulerSpec GlobalNorm(float guidance, float delta, float min, float max = 1f, float eps = 1e-8f)
        => new(guidance, delta, GuidanceForm.Extrapolate, GuidanceNorm.Global(min, max, eps));

    /// <summary>The per-row normalized step (Lumina 2): <c>uncond + g · (cond − uncond)</c>, rescaled per last-dimension row.</summary>
    public static GuidedEulerSpec RowNorm(float guidance, float delta, float eps = 1e-12f)
        => new(guidance, delta, GuidanceForm.Extrapolate, GuidanceNorm.PerRow(eps));
}
