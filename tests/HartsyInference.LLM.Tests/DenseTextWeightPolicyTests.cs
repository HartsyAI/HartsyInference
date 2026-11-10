using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.LLM.Tests;

/// <summary>The load dtype a dense text decoder picks for each stored format, on backends with different packed kernels. No weights.</summary>
public sealed class DenseTextWeightPolicyTests
{
    /// <summary>A backend that keeps exactly the given block types packed and refuses every other call.</summary>
    public class PackingBackend : DispatchProxy
    {
        private HashSet<DType> _packed = [];

        protected override object? Invoke(MethodInfo? method, object?[]? args) => method?.Name switch
        {
            "SupportsResidentQuant" when args?[0] is DType dtype => _packed.Contains(dtype),
            "SupportsResidentQuant" when args?[0] is Tensor t => _packed.Contains(t.DType),
            "get_Capabilities" => new BackendCapabilities { Name = "stub", SupportsF16 = true },
            _ => throw new NotSupportedException($"The packing stub has no {method?.Name}."),
        };

        public static IBackend Create(params DType[] packed)
        {
            IBackend backend = Create<IBackend, PackingBackend>();
            ((PackingBackend)(object)backend)._packed = [.. packed];
            return backend;
        }
    }

    private static HfQuantWeight Weight(HfWeightStorage storage, DType stored, long cols = 4096) =>
        new("model.layers.0.mlp.up_proj.weight", new TensorShape(128, cols), storage, stored, "k");

    [Theory]
    [InlineData(HfWeightStorage.Nf4, "Q4_0")]
    [InlineData(HfWeightStorage.GptqInt4, "Q4_0")]
    [InlineData(HfWeightStorage.Fp8Channel, "Q8_0")]
    [InlineData(HfWeightStorage.Fp8Block, "Q8_0")]
    public void QuantizedProjectionsStayPackedAtTheirWidth(HfWeightStorage storage, string expected)
    {
        IBackend vulkanLike = PackingBackend.Create(DType.Q4_0, DType.Q8_0);
        Assert.Equal(expected, DenseTextWeightPolicy.Target(Weight(storage, DType.U8), projection: true, vulkanLike).Name);
    }

    [Fact]
    public void AFourBitSourcePrefersQ4KWhereItFitsAndAnExplicitTargetWins()
    {
        IBackend vulkanLike = PackingBackend.Create(DType.Q4_0, DType.Q4_K, DType.Q8_0);
        HfQuantWeight nf4 = Weight(HfWeightStorage.Nf4, DType.U8);
        Assert.Equal(DType.Q4_K, DenseTextWeightPolicy.Target(nf4, true, vulkanLike));
        // A row that is not whole 256-value super-blocks falls back to Q4_0.
        Assert.Equal(DType.Q4_0, DenseTextWeightPolicy.Target(Weight(HfWeightStorage.Nf4, DType.U8, cols: 4096 + 32), true, vulkanLike));
        Assert.Equal(DType.Q8_0, DenseTextWeightPolicy.Target(nf4, true, vulkanLike, quantizedProjections: DType.Q8_0));
    }

    [Fact]
    public void AFourBitSourceFallsBackToQ8BeforeWideningWhenQ4IsMissing()
    {
        IBackend q8Only = PackingBackend.Create(DType.Q8_0);
        Assert.Equal(DType.Q8_0, DenseTextWeightPolicy.Target(Weight(HfWeightStorage.Nf4, DType.U8), true, q8Only));
        Assert.Equal(DType.F16, DenseTextWeightPolicy.Target(Weight(HfWeightStorage.Nf4, DType.U8), true, PackingBackend.Create()));
    }

    [Fact]
    public void TheHostAndCpuGetF32AndNonProjectionsAreNeverPacked()
    {
        using CpuBackend cpu = new();
        Assert.Equal(DType.F32, DenseTextWeightPolicy.Target(Weight(HfWeightStorage.Nf4, DType.U8), true, null));
        Assert.Equal(DType.F32, DenseTextWeightPolicy.Target(Weight(HfWeightStorage.Fp8Channel, DType.F8E4M3), true, cpu));
        IBackend vulkanLike = PackingBackend.Create(DType.Q4_0, DType.Q8_0);
        Assert.Equal(DType.F32, DenseTextWeightPolicy.Target(Weight(HfWeightStorage.Fp8Channel, DType.F8E4M3), projection: false, vulkanLike));
    }

    [Fact]
    public void PlainProjectionsKeepTheirDtypeUnlessAskedAndPackable()
    {
        IBackend vulkanLike = PackingBackend.Create(DType.Q4_0, DType.Q8_0);
        HfQuantWeight bf16 = Weight(HfWeightStorage.Plain, DType.BF16);
        Assert.Equal(DType.BF16, DenseTextWeightPolicy.Target(bf16, true, vulkanLike));
        Assert.Equal(DType.Q8_0, DenseTextWeightPolicy.Target(bf16, true, vulkanLike, DType.Q8_0));
        Assert.Equal(DType.BF16, DenseTextWeightPolicy.Target(bf16, true, vulkanLike, DType.Q6_K));   // not packable here
        Assert.Equal(DType.BF16, DenseTextWeightPolicy.Target(bf16, projection: false, vulkanLike, DType.Q8_0));
        Assert.Equal(DType.BF16, DenseTextWeightPolicy.Target(Weight(HfWeightStorage.Plain, DType.BF16, cols: 100), true, vulkanLike, DType.Q8_0));
    }

    [Theory]
    [InlineData(null, null)]
    [InlineData("native", null)]
    [InlineData("q8_0", "Q8_0")]
    [InlineData("Q4_K", "Q4_K")]
    [InlineData("q4", "Q4_K")]      // a bare q4 means what it means for every other model
    [InlineData("q4_0", "Q4_0")]
    [InlineData("q6", "Q6_K")]
    public void PrecisionNamesParse(string? name, string? expected) =>
        Assert.Equal(expected, DenseTextWeightPolicy.ParsePlainPrecision(name)?.Name);

    [Fact]
    public void AnUnknownPrecisionIsRefusedByName() =>
        Assert.Contains("q3", Assert.Throws<ArgumentException>(() => DenseTextWeightPolicy.ParsePlainPrecision("q3")).Message);
}
