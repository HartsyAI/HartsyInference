using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

/// <summary>Holding the flow's block linears as F16 or BF16: which tensors narrow, and that a forward over them stays close to the F32 one.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class Trellis2FlowWeightDTypeTests
{
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void OnlyTheBlockMatricesNarrow(bool dense)
    {
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 21);
        using Trellis2FlowWeights weights = Trellis2FlowWeights.FromTensors(config, tensors, linearDType: DType.F16);

        foreach ((string key, Tensor t) in weights.Tensors)
        {
            bool blockMatrix = key.StartsWith("blocks.", StringComparison.Ordinal) && key.EndsWith(".weight", StringComparison.Ordinal) && t.Shape.Rank == 2;
            Assert.True(t.DType == (blockMatrix ? DType.F16 : DType.F32), $"{key} is {t.DType}");
        }

        Assert.Contains(weights.Tensors, e => e.Value.DType == DType.F16);
        Trellis2TestData.DisposeAll(tensors.Values);
    }

    [Fact]
    public void AnUnsupportedTypeIsRefused()
    {
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense: false);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 22);

        Assert.Throws<ArgumentOutOfRangeException>(() => Trellis2FlowWeights.FromTensors(config, tensors, linearDType: DType.I8));
        Trellis2TestData.DisposeAll(tensors.Values);
    }

    [Theory]
    [InlineData(true, 2e-3)]
    [InlineData(false, 2e-3)]
    public void AForwardOverF16WeightsStaysCloseToTheF32Forward(bool dense, double relativeL2)
    {
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 23);
        using Trellis2FlowWeights wide = Trellis2FlowWeights.FromTensors(config, tensors);
        using Trellis2FlowWeights narrow = Trellis2FlowWeights.FromTensors(config, tensors, linearDType: DType.F16);
        using CpuBackend backend = new();
        Random rng = new(24);

        float[] expected = Run(wide, config, dense, rng, backend);
        float[] actual = Run(narrow, config, dense, new Random(24), backend);

        double err = 0, reference = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            err += d * d;
            reference += expected[i] * (double)expected[i];
        }

        Assert.True(Math.Sqrt(err / reference) < relativeL2, $"relative L2 {Math.Sqrt(err / reference):E3}");
        Trellis2TestData.DisposeAll(tensors.Values);
    }

    private static float[] Run(Trellis2FlowWeights weights, Trellis2FlowConfig config, bool dense, Random rng, CpuBackend backend)
    {
        using Tensor cond = Trellis2TestData.Random(rng, 1, 5, config.CondChannels);
        if (dense)
        {
            int r = config.Resolution;
            using Trellis2SparseStructureFlow flow = new(weights);
            using Tensor latent = Trellis2TestData.Random(rng, 1, config.InChannels, r, r, r);
            using Tensor y = flow.Forward(backend, latent, 640f, cond);
            return Trellis2TestData.ToArray(y);
        }

        using Trellis2SlatFlow sparse = new(weights);
        int[] coords = [0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 1, 1, 0, 2, 2, 3];
        int n = coords.Length / 4;
        using Tensor x = Trellis2TestData.Random(rng, 1, n, config.InChannels);
        using Tensor z = sparse.Forward(backend, x, coords, 300f, cond);
        return Trellis2TestData.ToArray(z);
    }
}
