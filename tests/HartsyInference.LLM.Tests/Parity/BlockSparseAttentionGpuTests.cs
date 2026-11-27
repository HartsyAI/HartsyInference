using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>Vulkan has no block-sparse attention kernel: it says so and refuses the call by name, writing nothing. CUDA's kernel is
/// compared with the reference in HartsyInference.Core.Tests.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class BlockSparseAttentionGpuTests
{
    [VulkanFact]
    public void VulkanRefusesByName()
    {
        using VulkanBackend backend = new();
        RefusesByName(backend);
    }

    private static void RefusesByName(IBackend backend)
    {
        Assert.False(backend.HasBlockSparseAttention);
        TensorShape shape = new(1, 1, 64, 16);
        using Tensor query = new(shape, DType.F32), key = new(shape, DType.F32), value = new(shape, DType.F32);
        using Tensor output = new(shape, DType.F32);
        output.AsSpan<float>().Fill(-7f);
        BlockSparseAttentionOptions options = new() { QueryGrid = new(4, 4, 4), KeyGrid = new(4, 4, 4), Sparsity = 0.0 };

        NotSupportedException refusal = Assert.Throws<NotSupportedException>(
            () => backend.BlockSparseAttention(output, query, key, value, options));

        Assert.Contains(nameof(IBackend.BlockSparseAttention), refusal.Message);
        Assert.Contains(backend.Device.ToString(), refusal.Message);
        backend.Sync();
        Assert.All(output.AsReadOnlySpan<float>().ToArray(), static v => Assert.Equal(-7f, v));
    }
}
