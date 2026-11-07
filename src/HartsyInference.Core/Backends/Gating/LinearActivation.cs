namespace HartsyInference.Core.Backends.Gating;

/// <summary>The activation a <see cref="LinearFusion"/> applies to the projection's output.</summary>
public enum LinearActivation
{
    /// <summary>The projection as it is.</summary>
    None,

    /// <summary>The tanh GELU.</summary>
    GeluTanh,
}
