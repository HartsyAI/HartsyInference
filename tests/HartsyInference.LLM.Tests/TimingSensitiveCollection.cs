using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Runs test classes that hold code to a wall-clock budget apart from every other class, so a busy thread
/// pool or a parallel neighbour's GC cannot push them over it.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class TimingSensitiveCollection
{
    public const string Name = "timing-sensitive";
}
