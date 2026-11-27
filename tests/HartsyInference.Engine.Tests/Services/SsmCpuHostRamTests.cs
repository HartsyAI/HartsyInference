using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The host-RAM guard of an SSM load on the CPU: the expansion of every quantized weight to F32, read from the header. The 9B Q4_K_M
/// expands to about 36 GB, so the guard must refuse on that, not on 2.5x its 5.8 GB file.</summary>
public sealed class SsmCpuHostRamTests
{
    private static GgufTensorDescriptor Tensor(string name, DType dtype, int rows, int cols) => new()
    {
        Name = name, DType = dtype, Shape = new TensorShape(rows, cols), GgufTypeId = 0, RelativeOffset = 0,
    };

    [Fact]
    public void TheExpansionCountsEveryQuantizedTensorAtFourBytesPerElementAndNoOther()
    {
        const long file = 1_000;
        GgufTensorDescriptor[] tensors =
        [
            Tensor("blk.0.ffn_down.weight", DType.Q8_0, 4096, 4096),   // quantized: 4096 * 4096 * 4 bytes as F32
            Tensor("output_norm.weight", DType.F32, 1, 4096),          // already F32: not expanded
            Tensor("token_embd.weight", DType.F16, 248_320, 4096),     // a float, not quantized: not counted here
        ];
        Assert.Equal(file + 4096.0 * 4096 * 4, TextService.ExpandedF32RequiredBytes(tensors, file));
    }

    [Fact]
    public void ALoadWhoseExpansionExceedsTheFreeRamIsRefusedBeforeAnyAllocation()
    {
        const double gb = 1024.0 * 1024 * 1024;
        // The 9B expansion (about 36 GB) against 20 GB free: refused, and the message names the CPU expansion.
        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() =>
            TextService.RequireHostRam(availableBytes: 20 * gb, requiredBytes: 36 * gb, fileBytes: (long)(5.8 * gb), expandedBytes: 30 * gb,
                path: "SenseNova-Loop-9B-Q4_K_M.gguf", dequantizesEverything: false, expandQuantizedToF32: true));
        Assert.Contains("expanded to F32 for the CPU", ex.Message);
    }

    [Fact]
    public void ALoadWithEnoughFreeRamPasses()
    {
        const double gb = 1024.0 * 1024 * 1024;
        TextService.RequireHostRam(availableBytes: 50 * gb, requiredBytes: 36 * gb, fileBytes: (long)(5.8 * gb), expandedBytes: 30 * gb,
            path: "SenseNova-Loop-9B-Q4_K_M.gguf", dequantizesEverything: false, expandQuantizedToF32: true);
    }
}
