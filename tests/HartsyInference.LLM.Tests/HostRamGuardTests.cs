using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.LLM.Tests;

public sealed class HostRamGuardTests
{
    private static GgufTensorDescriptor Tensor(string name, DType type, int rows, int cols) => new()
    {
        Name = name, DType = type, Shape = new TensorShape(rows, cols), GgufTypeId = 0, RelativeOffset = 0,
    };

    [Fact]
    public void DequantizedHostBytes_CountsOnlyQuantizedTypesTheGpuCannotKeepCompressed()
    {
        GgufTensorDescriptor[] tensors =
        [
            Tensor("a", DType.Q4_K, 256, 256),
            Tensor("b", DType.Q6_K, 256, 256),
            Tensor("c", DType.F32, 256, 256),
            Tensor("d", DType.Q4_1, 256, 256),
        ];

        Assert.Equal(256.0 * 256 * 4, TextService.DequantizedHostBytes(tensors));
    }

    [Fact]
    public void DequantizedHostBytes_CountsTheEmbeddingTablesTheLoadAlwaysWidens()
    {
        GgufTensorDescriptor[] tensors =
        [
            Tensor("token_embd.weight", DType.Q4_K, 1024, 256),
            Tensor("per_layer_token_embd.weight", DType.Q6_K, 2048, 256),
            Tensor("blk.0.attn_q.weight", DType.Q4_K, 256, 256),
        ];

        Assert.Equal((1024.0 * 256 + 2048.0 * 256) * 4, TextService.DequantizedHostBytes(tensors));
    }

    [Fact]
    public void DequantizedHostBytes_WithABackendKeepSet_CountsEveryOtherQuantAndHalfFloat()
    {
        // Vulkan keeps a subset: Q3_K is packed on CUDA but widened there, and so are F16 tensors.
        GgufTensorDescriptor[] tensors =
        [
            Tensor("a", DType.Q4_K, 256, 256),
            Tensor("b", DType.Q3_K, 256, 256),
            Tensor("c", DType.F16, 256, 256),
            Tensor("d", DType.F32, 256, 256),
        ];
        HashSet<DType> keep = [DType.Q4_K, DType.Q6_K];

        Assert.Equal(0.0, TextService.DequantizedHostBytes(tensors));
        Assert.Equal(2 * 256.0 * 256 * 4, TextService.DequantizedHostBytes(tensors, keep));
    }
}
