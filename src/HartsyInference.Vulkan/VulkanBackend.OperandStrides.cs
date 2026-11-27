namespace HartsyInference.Vulkan;

// The element strides of one attention operand, and the push-constant block they are written to. The query-tiled kernel
// (VulkanBackend.FlashTiled.cs) and the cooperative-matrix kernels (VulkanBackend.FlashCm2.cs) pass Q, K, V and O to the shader this
// way, and the rows that choose between them (VulkanBackend.AttentionRows.cs) build the values.
public sealed partial class VulkanBackend
{
    /// <summary>Element strides of one tensor: between tokens (rows), between heads and between batch items. The innermost dim is contiguous.</summary>
    internal readonly record struct OperandStrides(uint Row, uint Head, uint Batch);

    /// <summary>Writes <paramref name="s"/> as three consecutive 32-bit values at <paramref name="offset"/> of a push-constant block.</summary>
    private static void WriteOperandStrides(Span<byte> pc, int offset, OperandStrides s)
    {
        BinaryWriteUInt(pc, offset, s.Row);
        BinaryWriteUInt(pc, offset + 4, s.Head);
        BinaryWriteUInt(pc, offset + 8, s.Batch);
    }
}
