using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>A block-scale companion used to be dropped silently, leaving the raw fp8 weight to run unscaled.</summary>
public sealed class Fp8BlockScaleGuardTests
{
    [Fact]
    public void Rank2Fp8Scale_ThrowsInsteadOfBeingDropped()
    {
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>
        {
            ["w.weight"] = new Tensor(new TensorShape(64, 64), DType.F8E4M3),
            ["w.weight_scale"] = new Tensor(new TensorShape(2, 2), DType.F32),
        };

        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => CheckpointConvertUtils.ApplyFp8ScaledDequant(weights));

        Assert.Contains("w.weight", ex.Message);
    }
}
