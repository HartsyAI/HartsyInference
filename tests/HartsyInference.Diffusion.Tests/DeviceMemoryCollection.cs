using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A test that reads the card's free memory counts whatever any other test allocates meanwhile. A collection that disables
/// parallelization keeps it alone.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class DeviceMemoryCollection
{
    public const string Name = "DeviceMemory";
}
