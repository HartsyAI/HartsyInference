using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Utonia;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Utonia;

public sealed class UtoniaWeightsTests
{
    [Fact]
    public void Get_RejectsNonF32TensorAndNamesTheKey()
    {
        using Tensor half = new(new TensorShape(2, 3), DType.F16);
        UtoniaWeights weights = new(new Dictionary<string, Tensor>(StringComparer.Ordinal)
        {
            ["Embedding.Linear.Weight"] = half,
        });

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => weights.Get("Embedding.Linear.Weight", 2, 3));

        Assert.Contains("Embedding.Linear.Weight", error.Message, StringComparison.Ordinal);
        Assert.Contains("F16", error.Message, StringComparison.Ordinal);
        Assert.Empty(weights.Consumed);
    }
}
