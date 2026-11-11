using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.LLM.Kimi;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Quant;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>A latent SiTU MoE with MXFP4-packed routed experts must match the same MoE with those experts dequantized to F32.</summary>
public sealed class PackedMoeTests
{
    private static readonly float[] E2M1 = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f];

    [Fact]
    public void PackedExperts_MatchTheF32Dequantization() => Compare(new CpuBackend());

    // The same MoE on a device backend: the packed projections run through IBackend.Mxfp4LinearF32 there.
    [Fact]
    public void Cuda_PackedExperts_MatchTheF32Dequantization()
    {
        if (!HartsyInference.Cuda.CudaContext.IsAvailable()) return;
        using HartsyInference.Cuda.CudaBackend backend = new(0, PtxDir());
        Compare(backend);
    }

    [Fact]
    public void Vulkan_PackedExperts_MatchTheF32Dequantization()
    {
        if (!HartsyInference.Vulkan.VulkanContext.IsAvailable()) return;
        using HartsyInference.Vulkan.VulkanBackend backend = new();
        Compare(backend);
    }

    private static void Compare(IBackend device)
    {
        KimiK31Config c = new()
        {
            Hidden = 16, Vocab = 20, Layers = 2, DenseLayers = 0, DenseIntermediate = 24, Heads = 2, KdaHeadDim = 8, KdaForgetRank = 4,
            FullAttentionLayers = new HashSet<int>(), QLoraRank = 8, KvLoraRank = 6, QkNopeDim = 4, QkRopeDim = 4, VDim = 4,
            IndexHeads = 2, IndexHeadDim = 8, IndexTopK = 16, Experts = 4, ExpertsPerToken = 2, MoeIntermediate = 32,
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
        using Tensor expected = reference.Forward(new CpuBackend(), input, tokens);
        using Tensor actual = packedMoe.Forward(device, input, tokens);

        float[] want = expected.AsReadOnlySpan<float>().ToArray();
        float[] got = actual.AsReadOnlySpan<float>().ToArray();
        for (int i = 0; i < want.Length; i++) Assert.Equal(want[i], got[i], 3);
    }

    // Random E2M1 nibbles and E8M0 scales for a [rows, cols] projection; returns the packed view and its exact F32 values.
    private static string PtxDir()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        return Path.Combine(AppContext.BaseDirectory, "Ptx");
    }

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
