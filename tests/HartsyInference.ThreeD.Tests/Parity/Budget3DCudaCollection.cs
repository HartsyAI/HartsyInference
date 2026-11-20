using Xunit;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>The Budget3D tests that use the CUDA card or <c>BudgetTimer</c>'s static state; xUnit runs the classes of one collection one at a time, so a profile never overlaps a parity run on the card or in the timer.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class Budget3DCudaCollection
{
    public const string Name = "Budget3D CUDA";
}
