using SharpInference.Core.Memory;
using Xunit;

namespace SharpInference.Core.Tests;

/// <summary>Tests for <see cref="NativeBuffer"/>.</summary>
public sealed unsafe class NativeBufferTests
{
    [Fact]
    public void Allocate_HasCorrectSize()
    {
        using NativeBuffer buffer = new NativeBuffer(1024);

        Assert.Equal((nuint)1024, buffer.ByteLength);
    }

    [Fact]
    public void Allocate_HasCorrectAlignment()
    {
        using NativeBuffer buffer = new NativeBuffer(1024);

        Assert.Equal((nuint)64, buffer.Alignment);
    }

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
    public void AsSpan_CorrectLength()
    {
        using NativeBuffer buffer = new NativeBuffer(32);

        Span<float> span = buffer.AsSpan<float>();

        Assert.Equal(8, span.Length);
    }

    [Fact]
    public void AsSpan_WriteAndRead()
    {
        using NativeBuffer buffer = new NativeBuffer(16);

        Span<float> writeSpan = buffer.AsSpan<float>();
        writeSpan[0] = 1.5f;
        writeSpan[1] = 2.5f;
        writeSpan[2] = 3.5f;
        writeSpan[3] = 4.5f;

        ReadOnlySpan<float> readSpan = buffer.AsReadOnlySpan<float>();
        Assert.Equal(1.5f, readSpan[0]);
        Assert.Equal(2.5f, readSpan[1]);
        Assert.Equal(3.5f, readSpan[2]);
        Assert.Equal(4.5f, readSpan[3]);
    }

    [Fact]
    public void Dispose_PointerAccess_Throws()
    {
        NativeBuffer buffer = new NativeBuffer(64);
        buffer.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
        {
            void* _ = buffer.Pointer;
        });
    }

    [Fact]
    public void Dispose_AsSpan_Throws()
    {
        NativeBuffer buffer = new NativeBuffer(64);
        buffer.Dispose();

        Assert.Throws<ObjectDisposedException>(() => buffer.AsSpan<byte>());
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
