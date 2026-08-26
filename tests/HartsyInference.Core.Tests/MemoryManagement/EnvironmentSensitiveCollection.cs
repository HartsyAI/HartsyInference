using Xunit;

namespace HartsyInference.Core.Tests.MemoryManagement;

/// <summary>Serializes the test classes that mutate process-global environment variables.</summary>
/// <remarks>xUnit runs test classes in parallel, and an environment variable is process-global, so two classes
/// that set <c>HARTSY_LOWVRAM</c> or <c>HARTSY_KEEP_MODELS</c> race: one clears the value the other just set and
/// asserts on, producing a failure that reproduces only under the full suite and passes in isolation.
/// <para>This is the flakiness that the move to typed configuration exists to remove — once nothing reads the
/// environment, these classes have no shared mutable state and the collection can go with them.</para></remarks>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class EnvironmentSensitiveCollection
{
    public const string Name = "environment-sensitive";
}
