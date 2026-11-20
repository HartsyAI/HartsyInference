using HartsyInference.Core.Backends.Gpu;
using HartsyInference.Core.Exceptions;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>The lifetime rule of a shared scratch buffer: nothing a live captured graph used is freed under it.</summary>
public sealed class HighWaterWorkspaceTests
{
    /// <summary>A device that hands out fresh addresses, records every call in order and can refuse a size.</summary>
    private sealed class FakeDevice
    {
        private ulong _next = 0x1000;
        public readonly HashSet<ulong> Live = [];
        public readonly List<string> Log = [];
        public long Capacity = long.MaxValue;

        public ulong Allocate(long bytes)
        {
            if (bytes > Capacity) throw new OutOfVramException($"{bytes} bytes do not fit.");
            ulong pointer = _next;
            _next += 0x1000;
            Live.Add(pointer);
            Log.Add($"alloc {bytes}");
            return pointer;
        }

        public void Free(ulong pointer)
        {
            Assert.True(Live.Remove(pointer), "A pointer was freed twice or never allocated.");
            Log.Add("free");
        }

        public HighWaterWorkspace Workspace() => new(Allocate, Free);
    }

    [Fact]
    public void ASmallerRequestReusesTheHighWaterBuffer()
    {
        FakeDevice device = new();
        HighWaterWorkspace workspace = device.Workspace();
        Assert.Equal(0UL, workspace.Use(64, lessee: null));
        ulong pointer = workspace.Grow(256);

        Assert.Equal(pointer, workspace.Use(256, lessee: null));
        Assert.Equal(pointer, workspace.Use(64, lessee: null));
        Assert.Equal(0UL, workspace.Use(257, lessee: null));
        Assert.Equal(256, workspace.Bytes);
        Assert.Equal(["alloc 256"], device.Log);
    }

    [Fact]
    public void AnUnleasedBufferIsFreedBeforeItsReplacementIsAllocated()
    {
        FakeDevice device = new();
        HighWaterWorkspace workspace = device.Workspace();
        workspace.Grow(256);

        workspace.Grow(1024);

        Assert.Equal(["alloc 256", "free", "alloc 1024"], device.Log);
        Assert.Single(device.Live);
        Assert.Equal(0, workspace.RetainedBytes);
    }

    [Fact]
    public void ABufferALiveGraphUsedOutlivesGrowth()
    {
        FakeDevice device = new();
        HighWaterWorkspace workspace = device.Workspace();
        object graph = new();
        ulong captured = workspace.Grow(256);
        Assert.Equal(captured, workspace.Use(128, graph));

        ulong grown = workspace.Grow(1024);

        Assert.NotEqual(captured, grown);
        Assert.Contains(captured, device.Live);
        Assert.Equal(256, workspace.RetainedBytes);
        Assert.Equal(["alloc 256", "alloc 1024"], device.Log);

        workspace.Release(graph);

        Assert.DoesNotContain(captured, device.Live);
        Assert.Contains(grown, device.Live);
        Assert.Equal(0, workspace.RetainedBytes);
        Assert.Equal(1024, workspace.Bytes);
    }

    [Fact]
    public void ABufferTwoGraphsUsedIsFreedAfterBothRelease()
    {
        FakeDevice device = new();
        HighWaterWorkspace workspace = device.Workspace();
        object first = new(), second = new();
        ulong captured = workspace.Grow(256);
        workspace.Use(256, first);
        workspace.Use(256, second);
        workspace.Use(256, second);
        workspace.Grow(1024);

        workspace.Release(first);
        Assert.Contains(captured, device.Live);
        workspace.Release(second);

        Assert.DoesNotContain(captured, device.Live);
        workspace.Release(second);
    }

    [Fact]
    public void ALeasedBufferStaysCurrentWhenItsReplacementDoesNotFit()
    {
        FakeDevice device = new() { Capacity = 512 };
        HighWaterWorkspace workspace = device.Workspace();
        object graph = new();
        ulong captured = workspace.Grow(256);
        workspace.Use(256, graph);

        Assert.Throws<OutOfVramException>(() => workspace.Grow(1024));

        Assert.Equal(captured, workspace.Use(256, lessee: null));
        Assert.Equal(256, workspace.Bytes);
        Assert.Equal(1024, workspace.GrowthFailedAt);
        Assert.Equal(0UL, workspace.Grow(2048));
        Assert.Equal(["alloc 256"], device.Log);
    }

    [Fact]
    public void AnUnleasedBufferIsGoneWhenItsReplacementDoesNotFit()
    {
        FakeDevice device = new() { Capacity = 512 };
        HighWaterWorkspace workspace = device.Workspace();
        workspace.Grow(256);

        Assert.Throws<OutOfVramException>(() => workspace.Grow(1024));

        Assert.Equal(0, workspace.Bytes);
        Assert.Empty(device.Live);
        // The failed size is not retried until the owner releases; a smaller one still allocates.
        Assert.Equal(0UL, workspace.Grow(1024));
        Assert.NotEqual(0UL, workspace.Grow(256));
    }

    [Fact]
    public void ReleasingTheCurrentBufferKeepsItWhileAGraphLeasesIt()
    {
        FakeDevice device = new() { Capacity = 512 };
        HighWaterWorkspace workspace = device.Workspace();
        object graph = new();
        ulong captured = workspace.Grow(256);
        workspace.Use(256, graph);
        Assert.Throws<OutOfVramException>(() => workspace.Grow(1024));

        workspace.ReleaseCurrent();

        Assert.Equal(0, workspace.Bytes);
        Assert.Equal(0UL, workspace.Use(64, lessee: null));
        Assert.Contains(captured, device.Live);
        Assert.Equal(256, workspace.RetainedBytes);
        Assert.Equal(long.MaxValue, workspace.GrowthFailedAt);

        workspace.Release(graph);

        Assert.Empty(device.Live);
    }

    [Fact]
    public void ReleasingTheCurrentBufferFreesItWhenNoGraphLeasesIt()
    {
        FakeDevice device = new();
        HighWaterWorkspace workspace = device.Workspace();
        workspace.Grow(256);

        workspace.ReleaseCurrent();
        workspace.ReleaseCurrent();

        Assert.Empty(device.Live);
        Assert.Equal(0, workspace.Bytes);
    }

    [Fact]
    public void DetachHandsEveryBufferBackAndFreesNone()
    {
        FakeDevice device = new();
        HighWaterWorkspace workspace = device.Workspace();
        object graph = new();
        ulong captured = workspace.Grow(256);
        workspace.Use(256, graph);
        ulong grown = workspace.Grow(1024);

        ulong[] pointers = workspace.Detach();

        Assert.Equal([grown, captured], pointers);
        Assert.Equal(2, device.Live.Count);
        Assert.Equal(0, workspace.Bytes);
        Assert.Equal(0, workspace.RetainedBytes);
        // The owner frees them itself; a late release from the graph must not free again.
        workspace.Release(graph);
        Assert.Equal(2, device.Live.Count);
    }
}
