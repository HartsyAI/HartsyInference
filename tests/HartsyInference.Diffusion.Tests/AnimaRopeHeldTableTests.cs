using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>AnimaRope holds one device-table lease per backend. A grid change releases the held lease; when that release throws,
/// the backend must not keep an entry pointing at tables a later retry frees.</summary>
public sealed unsafe class AnimaRopeHeldTableTests
{
    [Fact]
    public void GridChange_WhoseReleaseThrows_LeavesNoStaleEntry()
    {
        AnimaRope rope = new(24, 10000f, (1f, 1f, 1f));
        RopeRecordingBackend a = new();
        RopeRecordingBackend b = new();
        rope.GetOrCreateTables(a, 1, 2, 2);
        // b's grid evicts a's (one signature is retained), so a's tables are retired but still under a's lease.
        rope.GetOrCreateTables(b, 1, 3, 3);

        a.FailFree = true;
        Assert.Throws<InvalidOperationException>(() => rope.GetOrCreateTables(a, 1, 3, 3));
        a.FailFree = false;
        rope.ReleaseGpuTables(b);   // a later flush frees a's first tables
        Assert.Contains("free:2", a.Calls);

        (Tensor cos, Tensor sin) = rope.GetOrCreateTables(a, 1, 2, 2);

        _ = cos.DataPointer;   // a freed table would throw ObjectDisposedException here
        _ = sin.DataPointer;
        rope.ReleaseGpuTables(a);
    }
}
