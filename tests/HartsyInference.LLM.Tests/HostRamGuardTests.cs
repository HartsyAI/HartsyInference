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
            Tensor("d", DType.Q3_K, 256, 256),
        ];

        Assert.Equal(256.0 * 256 * 4, TextService.DequantizedHostBytes(tensors));
    }

    [Fact]
    public void DequantizedHostBytes_IsZeroWhenEveryQuantStaysCompressed()
    {
        Assert.Equal(0, TextService.DequantizedHostBytes([Tensor("a", DType.Q4_K, 256, 256), Tensor("b", DType.Q8_0, 256, 256)]));
    }
}
