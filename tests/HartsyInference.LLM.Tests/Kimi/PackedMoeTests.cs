using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.LLM.Kimi;
using HartsyInference.LLM.Tests.Parity;
using HartsyInference.LLM.Tests.Quant;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Quant;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>A latent SiTU MoE with MXFP4-packed routed experts must match the same MoE with those experts dequantized to F32 on the
/// CPU backend. The F32 lanes bound each element's difference over the larger of 1 and the reference value; CUDA's TF32 lane is
/// bounded as the attention contract bounds TF32.</summary>
public sealed class PackedMoeTests(ITestOutputHelper output)
{
    /// <summary>The bound where every product is F32 and a backend differs from the reference by rounding order only. Largest
    /// differences measured at <c>7679bde01</c> on an RTX 4090 host: CUDA with exact GEMMs 5.4e-7, Vulkan 5.0e-7, CPU 2.4e-7.</summary>
    private const float ExactTolerance = 1e-5f;

    /// <summary>CUDA's default runs the router, latent and shared-expert GEMMs in TF32; the packed projections are an F32 kernel
    /// there too. This is the TF32 figure of <c>docs/ATTENTION_CONTRACT.md</c> §8: a bound on the relative L2, paired with an absolute
    /// one of this figure times the largest |reference|, which the test holds for each element. Measured on an RTX 4090 at
    /// <c>7679bde01</c>: relative L2 6.9e-4; the largest difference over max(1, |reference|) 2.7e-3, at element 33.</summary>
    private const float Tf32Tolerance = 2e-3f;

    private static readonly float[] E2M1 = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f];

    [Fact]
    public void PackedExperts_MatchTheF32Dequantization()
    {
        using CpuBackend cpu = new();
        AssertWithinRounding(Measure(cpu));
    }

    // The same MoE on a device backend: the packed projections run through IBackend.Mxfp4LinearF32 there.
    [CudaFact]
    [Trait("Category", "GpuIntegration")]
    public void Cuda_PackedExperts_MatchTheF32Dequantization_ExactGemms()
    {
        using HartsyInference.Cuda.CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        PackedMxfp4CudaTests.RequireDeviceKernel(backend);
        using ExactGemmScope exact = new(backend);
        AssertWithinRounding(Measure(backend));
    }

    [CudaFact]
    [Trait("Category", "GpuIntegration")]
    public void Cuda_PackedExperts_MatchTheF32Dequantization()
    {
        using HartsyInference.Cuda.CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        PackedMxfp4CudaTests.RequireDeviceKernel(backend);
        Difference found = Measure(backend);
        Assert.True(found.RelativeL2 <= Tf32Tolerance, $"{found.Summary}; the relative L2 bound is {Tf32Tolerance:G2}.");
        float absolute = Tf32Tolerance * found.LargestReference;
        Assert.True(found.LargestAbsolute <= absolute, $"{found.Summary}; the absolute bound is {absolute:G3}.");
    }

    [VulkanFact]
    [Trait("Category", "GpuIntegration")]
    public void Vulkan_PackedExperts_MatchTheF32Dequantization()
    {
        using HartsyInference.Vulkan.VulkanBackend backend = new();
        AssertWithinRounding(Measure(backend));
    }

    private static void AssertWithinRounding(Difference found) =>
        Assert.True(found.LargestRelative <= ExactTolerance, $"{found.Summary}; the bound is {ExactTolerance:G2}.");

    /// <summary>How far a device's output is from the reference. A NaN on either side makes each figure it reaches NaN, which is
    /// within no bound.</summary>
    private readonly record struct Difference(float LargestRelative, float LargestAbsolute, float LargestReference, double RelativeL2,
        string Summary);

    // The MoE with F32 experts on the CPU against the one with packed experts on the device.
    private Difference Measure(IBackend device)
    {
        KimiK31Config c = new()
        {
            Hidden = 16, Vocab = 20, Layers = 2, DenseLayers = 0, DenseIntermediate = 24, Heads = 2, KdaHeadDim = 8, KdaForgetRank = 4,
            FullAttentionLayers = new HashSet<int>(), QLoraRank = 8, KvLoraRank = 6, QkNopeDim = 4, QkRopeDim = 4, VDim = 4,
            IndexHeads = 2, IndexHeadDim = 8, IndexTopK = 16, Experts = 4, ExpertsPerToken = 2, MoeIntermediate = 64,
            SharedExperts = 1, LatentSize = 32, AttnResBlock = 2,
        };
        Random rng = new(21);
        Dictionary<string, Tensor> f32 = KimiK31ModelTests.MoeWeights(rng, c, "L0");
        string mlp = "L0.mlp";

        List<PackedMxfp4Expert> packed = new();
        for (int e = 0; e < c.Experts; e++)
        {
            PackedMxfp4Projection gate = Pack(rng, c.MoeIntermediate, c.LatentSize, out float[] gateW);
            PackedMxfp4Projection up = Pack(rng, c.MoeIntermediate, c.LatentSize, out float[] upW);
            PackedMxfp4Projection down = Pack(rng, c.LatentSize, c.MoeIntermediate, out float[] downW);
            packed.Add(new PackedMxfp4Expert(gate, up, down));
            f32[$"{mlp}.experts.{e}.gate_proj.weight"] = Floats(gateW, c.MoeIntermediate, c.LatentSize);
            f32[$"{mlp}.experts.{e}.up_proj.weight"] = Floats(upW, c.MoeIntermediate, c.LatentSize);
            f32[$"{mlp}.experts.{e}.down_proj.weight"] = Floats(downW, c.LatentSize, c.MoeIntermediate);
        }

        MoeFeedForward reference = new(KimiK31Model.MoeSettings(c), c.Hidden, lowVram: false);
        reference.LoadWeights(f32, "L0");

        Dictionary<string, Tensor> withoutExperts = f32.Where(kv => !kv.Key.Contains(".experts.", StringComparison.Ordinal))
            .ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal);
        MoeFeedForward packedMoe = new(KimiK31Model.MoeSettings(c), c.Hidden, lowVram: false);
        packedMoe.LoadWeights(withoutExperts, "L0");
        packedMoe.LoadPackedExperts(packed);

        const int tokens = 3;
        float[] x = new float[tokens * c.Hidden];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        using Tensor input = Floats(x, 1, tokens, c.Hidden);
        using CpuBackend cpu = new();
        using Tensor expected = reference.Forward(cpu, input, tokens);
        using Tensor actual = packedMoe.Forward(device, input, tokens);

        float[] want = expected.AsReadOnlySpan<float>().ToArray();
        float[] got = actual.AsReadOnlySpan<float>().ToArray();
        Assert.Equal(want.Length, got.Length);
        float worst = 0f, absolute = 0f;
        int at = 0;
        double squaredDifference = 0, squaredReference = 0;
        for (int i = 0; i < want.Length; i++)
        {
            squaredDifference += ((double)want[i] - got[i]) * ((double)want[i] - got[i]);
            squaredReference += (double)want[i] * want[i];
            float apart = MathF.Abs(want[i] - got[i]);
            float difference = apart / MathF.Max(1f, MathF.Abs(want[i]));
            // MathF.Max keeps a NaN; the index needs the explicit test, because a NaN compares false with everything.
            absolute = MathF.Max(absolute, apart);
            if (!float.IsNaN(worst) && (float.IsNaN(difference) || difference > worst)) (worst, at) = (difference, i);
        }
        float largestReference = want.Max(MathF.Abs);
        double relativeL2 = Math.Sqrt(squaredDifference / squaredReference);
        string summary = $"largest relative difference {worst:G6} at element {at}: reference {want[at]}, device {got[at]}, "
            + $"largest absolute difference {absolute:G6}, largest |reference| {largestReference:G4}, "
            + $"reference L2 norm {Math.Sqrt(squaredReference):G6}, relative L2 {relativeL2:G6}";
        output.WriteLine($"{device.GetType().Name}: {summary}");
        return new Difference(worst, absolute, largestReference, relativeL2, summary);
    }

    // Random E2M1 nibbles and E8M0 scales for a [rows, cols] projection; returns the packed view and its exact F32 values.
    private static PackedMxfp4Projection Pack(Random rng, int rows, int cols, out float[] values)
    {
        byte[] nibbles = new byte[rows * cols / 2];
        rng.NextBytes(nibbles);
        int blocks = cols / 32;
        byte[] scales = new byte[rows * blocks];
        for (int i = 0; i < scales.Length; i++) scales[i] = (byte)rng.Next(118, 126);
        values = new float[rows * cols];
        for (int o = 0; o < rows; o++)
            for (int c = 0; c < cols; c++)
            {
                int b = nibbles[o * (cols / 2) + c / 2];
                int nib = (c & 1) == 0 ? b & 0xF : b >> 4;
                float mag = E2M1[nib & 7] * (float)Math.Pow(2, scales[o * blocks + c / 32] - 127);
                values[o * cols + c] = (nib & 8) != 0 ? -mag : mag;
            }
        Tensor packed = new(new TensorShape(rows, cols / 2), DType.U8);
        nibbles.CopyTo(packed.AsSpan<byte>());
        Tensor scale = new(new TensorShape(rows, blocks), DType.U8);
        scales.CopyTo(scale.AsSpan<byte>());
        QuantRecipe recipe = new()
        {
            Encoding = QuantEncoding.Mxfp4E8M0, Geometry = new BlockGeometry(1, 32), ScaleDType = DType.U8,
            LogicalRows = rows, LogicalCols = cols, Scale = scale,
        };
        return new PackedMxfp4Projection(packed, recipe);
    }

    private static Tensor Floats(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
