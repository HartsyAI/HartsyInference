using HartsyInference.Core.Memory;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>Tests for <see cref="NativeBuffer"/>.</summary>
public sealed unsafe class NativeBufferTests
{
    [Fact]
    public void Allocate_ZeroBytes_Throws()
    {
        Assert.Throws<ArgumentException>(() => new NativeBuffer(0));
    }

    [Fact]
    public void Allocate_IsZeroed()
    {
        using NativeBuffer buffer = new NativeBuffer(256);

        Span<byte> span = buffer.AsSpan<byte>();
        for (int i = 0; i < span.Length; i++)
        {
            Assert.Equal(0, span[i]);
        }
    }

    [Fact]
    public void Dispose_Idempotent()
    {
        NativeBuffer buffer = new NativeBuffer(64);

        Exception? exception = Record.Exception(() =>
        {
            buffer.Dispose();
            buffer.Dispose();
        });

        Assert.Null(exception);
    }

    [Fact]
    public void Alignment_Custom_Respected()
    {
        using NativeBuffer buffer = new NativeBuffer(256, 128);

        Assert.Equal((nuint)128, buffer.Alignment);
    }
}
