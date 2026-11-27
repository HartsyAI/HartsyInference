using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.Core.Tests.Gemm;

/// <summary>Pins the batched cuBLAS dtype plan: the C type must follow the output dtype, and unsupported pairs must refuse.</summary>
public sealed class CublasGemmPlanTests
{
    /// <summary>(a, b, output, GEMM dtype) with default precision flags. Output is supported when it equals the GEMM dtype or is F32.</summary>
    public static TheoryData<string, string, string, string> SupportedCombinations => new()
    {
        { "F32", "F32", "F32", "F32" },
        { "F32", "F16", "F16", "F16" },
        { "F32", "F16", "F32", "F16" },
        { "F16", "F32", "F16", "F16" },
        { "F16", "F32", "F32", "F16" },
        { "F16", "F16", "F16", "F16" },
        { "F16", "F16", "F32", "F16" },
        { "F16", "BF16", "F16", "F16" },
        { "F16", "BF16", "F32", "F16" },
        { "BF16", "BF16", "BF16", "BF16" },
        { "BF16", "BF16", "F32", "BF16" },
        { "F32", "BF16", "BF16", "BF16" },
        { "BF16", "F32", "BF16", "BF16" },
        { "BF16", "F32", "F32", "BF16" },
        { "F8E4M3", "F32", "F32", "BF16" },
        { "F8E4M3", "F8E4M3", "F16", "F16" },
    };

    /// <summary>(a, b, output) pairs with no cuBLAS kernel: a float GEMM into a narrower or mismatched output, or a non-float dtype.</summary>
    public static TheoryData<string, string, string> UnsupportedCombinations => new()
    {
        { "F32", "F32", "F16" },
        { "F32", "F32", "BF16" },
        { "F16", "F16", "BF16" },
        { "BF16", "BF16", "F16" },
        { "F32", "F16", "BF16" },
        { "F32", "BF16", "F16" },
        { "F16", "BF16", "BF16" },
        { "F8E4M3", "F32", "F16" },
        { "F32", "F32", "I8" },
        { "I8", "I8", "F32" },
    };

    [Theory]
    [MemberData(nameof(SupportedCombinations))]
    public void Resolve_SupportedCombination_PlansGemmAndWritesRequestedOutput(string a, string b, string output, string gemm)
    {
        CublasGemmPlan plan = CublasGemmPlan.Resolve(Parse(a), Parse(b), Parse(output), false, false, false);

        Assert.Equal(Parse(gemm), plan.GemmDtype);
        Assert.Equal(Parse(output), plan.OutputDtype);
    }

    [Theory]
    [MemberData(nameof(UnsupportedCombinations))]
    public void Resolve_UnsupportedCombination_ThrowsNamingOperandsAndOutput(string a, string b, string output)
    {
        NotSupportedException ex = Assert.Throws<NotSupportedException>(
            () => CublasGemmPlan.Resolve(Parse(a), Parse(b), Parse(output), false, false, false));

        Assert.Contains($"operands {Parse(a).Name} x {Parse(b).Name} with output {Parse(output).Name} ", ex.Message);
    }

    [Fact]
    public void Resolve_EveryFloatCombination_EitherWritesRequestedOutputOrRefuses()
    {
        DType[] floats = [DType.F32, DType.F16, DType.BF16];
        int supported = 0;
        foreach (DType a in floats)
        foreach (DType b in floats)
        foreach (DType output in floats)
        {
            if (!TryResolve(a, b, output, out CublasGemmPlan plan)) continue;
            Assert.Equal(output, plan.OutputDtype);
            supported++;
        }

        // 9 operand pairs: F32xF32 has 1 output, every other pair resolves to an F16 or BF16 GEMM with 2.
        Assert.Equal(17, supported);
    }

    [Fact]
    public void Resolve_HighPrecisionMixedBf16_ComputesF32AndRefusesBf16Output()
    {
        CublasGemmPlan f32Out = CublasGemmPlan.Resolve(DType.BF16, DType.F32, DType.F32, true, false, false);
        Assert.Equal(DType.F32, f32Out.GemmDtype);
        Assert.Equal(DType.F32, f32Out.OutputDtype);

        Assert.Throws<NotSupportedException>(() => CublasGemmPlan.Resolve(DType.BF16, DType.F32, DType.BF16, true, false, false));
    }

    [Fact]
    public void Resolve_Fp8Operand_FollowsPrecisionFlags()
    {
        Assert.Equal(DType.BF16, CublasGemmPlan.Resolve(DType.F8E4M3, DType.F32, DType.F32, false, false, false).GemmDtype);
        Assert.Equal(DType.F32, CublasGemmPlan.Resolve(DType.F8E4M3, DType.F32, DType.F32, false, true, false).GemmDtype);
        Assert.Equal(DType.F16, CublasGemmPlan.Resolve(DType.F8E4M3, DType.F32, DType.F32, false, false, true).GemmDtype);
    }

    private static bool TryResolve(DType a, DType b, DType output, out CublasGemmPlan plan)
    {
        try
        {
            plan = CublasGemmPlan.Resolve(a, b, output, false, false, false);
            return true;
        }
        catch (NotSupportedException)
        {
            plan = default;
            return false;
        }
    }

    private static DType Parse(string name) => name switch
    {
        "F32" => DType.F32,
        "F16" => DType.F16,
        "BF16" => DType.BF16,
        "F8E4M3" => DType.F8E4M3,
        "I8" => DType.I8,
        _ => throw new ArgumentException($"Unknown dtype name {name}.", nameof(name)),
    };
}
