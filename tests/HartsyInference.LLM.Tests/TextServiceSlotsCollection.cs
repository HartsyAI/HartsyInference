using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Runs test classes that load and replace models through a <c>TextService</c> apart from every other class: a replaced model's forced collection pauses the
/// whole process, which would push a parallel neighbour over its wall-clock budget, and a knob such a class sets is read by any neighbour's model load.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class TextServiceSlotsCollection
{
    public const string Name = "text-service-slots";
}
