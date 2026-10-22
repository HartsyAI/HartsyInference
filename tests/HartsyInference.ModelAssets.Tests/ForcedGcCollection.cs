using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Runs the classes that force collections in a loop apart from every other class.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class ForcedGcCollection
{
    public const string Name = "forced-gc";
}
