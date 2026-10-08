using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe;

/// <summary>Where the selection bias is added. Production flat sigmoid routing biases the logit; grouped routing biases the score.</summary>
public enum SelectionBiasSpace
{
    /// <summary>Bias added to the scored value (sigmoid or softmax output), as <see cref="IBackend.MoeRoute"/> does.</summary>
    Score,

    /// <summary>Bias added to the raw logit before sigmoid scoring. Flat routers only.</summary>
    Logit,
}
