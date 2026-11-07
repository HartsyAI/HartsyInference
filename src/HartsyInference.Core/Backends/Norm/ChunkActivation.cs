namespace HartsyInference.Core.Backends.Norm;

/// <summary>The elementwise function <see cref="IBackend.ChunkActivate"/> applies to one chunk.</summary>
public enum ChunkActivation
{
    /// <summary><c>x</c>.</summary>
    Identity,

    /// <summary><c>1 + x</c>: an adaLN scale around identity.</summary>
    OnePlus,

    /// <summary><c>tanh(x)</c>: a bounded gate.</summary>
    Tanh,

    /// <summary><c>sigmoid(x)</c>.</summary>
    Sigmoid,
}
