using System.Runtime.InteropServices;

namespace HartsyInference.Vulkan.Authoring;

/// <summary>Builds a push-constant block by appending, so the byte offsets are not written by hand.
///
/// <para>The offsets were the problem. An op wrote <c>stackalloc byte[9 * 4]</c> and then nine
/// <c>BinaryWriteUInt(pc, 0/4/8/…)</c> calls whose offsets had to agree with a GLSL struct in another file and
/// with each other. Inserting a field meant renumbering every line below it, and getting one wrong does not fail —
/// the shader reads a plausible number from the wrong place.</para></summary>
/// <remarks>A MUTATING ref struct, deliberately not a fluent chain: chaining by value on a ref struct would copy
/// it, and each call would write at the offset the previous copy had, silently producing a block where every field
/// landed at zero.</remarks>
public ref struct PushConstants
{
    private readonly Span<byte> _buffer;
    private int _written;

    /// <summary>Wraps a caller-owned span, normally a <c>stackalloc byte[128]</c>.</summary>
    public PushConstants(Span<byte> buffer)
    {
        _buffer = buffer;
        _written = 0;
    }

    /// <summary>Bytes appended so far — what to hand to the dispatch.</summary>
    public readonly ReadOnlySpan<byte> Written => _buffer[.._written];

    /// <summary>Appends a 32-bit unsigned integer.</summary>
    public void U32(uint value) => Append(value, 4);

    /// <summary>Appends a signed 32-bit integer, as GLSL's <c>int</c>.</summary>
    public void I32(int value) => Append(value, 4);

    /// <summary>Appends a 32-bit float.</summary>
    public void F32(float value) => Append(value, 4);

    /// <summary>Appends a GLSL <c>bool</c>, which occupies a full 32-bit word.</summary>
    public void Bool(bool value) => Append(value ? 1u : 0u, 4);

    /// <summary>Appends a 64-bit unsigned integer, 8-byte aligned as the standard layout requires.</summary>
    /// <remarks>The alignment is not optional and is exactly the kind of rule a hand-written offset gets wrong: a
    /// <c>uint64_t</c> after an odd number of 32-bit fields must skip four bytes, and a shader reading a
    /// misaligned one gets two halves of neighbouring fields.</remarks>
    public void U64(ulong value)
    {
        if (_written % 8 != 0)
        {
            _written += 4;
        }
        Append(value, 8);
    }

    private void Append<T>(T value, int size) where T : unmanaged
    {
        if (_written + size > _buffer.Length)
        {
            throw new InvalidOperationException(
                $"Push-constant block overflows {_buffer.Length} bytes at offset {_written} (+{size}). "
                + $"Vulkan guarantees only {VulkanDescriptorManager.PushConstantRangeBytes} bytes; anything larger "
                + "belongs in a storage buffer.");
        }
        MemoryMarshal.Write(_buffer[_written..], in value);
        _written += size;
    }
}
