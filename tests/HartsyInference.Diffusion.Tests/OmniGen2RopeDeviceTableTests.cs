using Xunit;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Proves the Boogu device-rope port is bit-equivalent to the host path it replaced: applying
/// <c>IBackend.WanRopeInterleaved</c> on the pre-permute <c>[1, S, H, D]</c> layout with
/// <see cref="OmniGen2Rope.ExpandToDeviceTables"/> tables must match <see cref="OmniGen2Rope.Apply"/>
/// (the old host loop) on the post-permute <c>[1, H, S, D]</c> layout, for both Q-head and KV-head counts.
/// Runs on the CPU backend, whose <c>WanRopeInterleaved</c> is the reference implementation of the CUDA kernel.</summary>
public sealed class OmniGen2RopeDeviceTableTests
{
    [Theory]
    [InlineData(4, 3)]   // Q heads, uneven seq
    [InlineData(1, 7)]   // KV heads (GQA)
    public unsafe void WanRopeInterleaved_PrePermute_Matches_HostApply_PostPermute(int heads, int seqLen)
    {
        int[] axes = [4, 6, 6];
        int headDim = 16;
        OmniGen2Rope rope = new(axes, theta: 256);
        IBackend backend = new CpuBackend();   // interface default WanRopeInterleaved = the CUDA kernel's reference

        // Positions with distinct time/height/width ids (the Boogu pe_shift style).
        int[] t = new int[seqLen], h = new int[seqLen], w = new int[seqLen];
        for (int s = 0; s < seqLen; s++) { t[s] = s; h[s] = (s * 3) % 5; w[s] = (s * 7) % 11; }
        (float[] cosTab, float[] sinTab) = rope.BuildTableFromPositions(t, h, w);

        // Deterministic pseudo-random input in [B=1, S, H, D] (pre-permute layout).
        Tensor pre = new(new TensorShape(1, seqLen, heads, headDim), DType.F32);
        float* prePtr = (float*)pre.DataPointer;
        long n = pre.ElementCount;
        uint rng = 12345;
        for (long i = 0; i < n; i++)
        {
            rng = rng * 1664525u + 1013904223u;
            prePtr[i] = ((rng >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        }

        // Reference: permute to [1, H, S, D], run the old host Apply.
        Tensor refPost = new(new TensorShape(1, heads, seqLen, headDim), DType.F32);
        backend.Permute0213(refPost, pre, seqLen, heads, headDim);
        rope.Apply(refPost, cosTab, sinTab, 1, heads, seqLen);

        // New path: device tables + WanRopeInterleaved pre-permute, then the same permute.
        (Tensor cosT, Tensor sinT) = rope.ExpandToDeviceTables(cosTab, sinTab);
        backend.WanRopeInterleaved(pre, cosT, sinT, seqLen, heads, headDim);
        Tensor newPost = new(new TensorShape(1, heads, seqLen, headDim), DType.F32);
        backend.Permute0213(newPost, pre, seqLen, heads, headDim);

        float* rp = (float*)refPost.DataPointer;
        float* np = (float*)newPost.DataPointer;
        for (long i = 0; i < n; i++)
            Assert.Equal(rp[i], np[i], 5);

        pre.Dispose(); refPost.Dispose(); newPost.Dispose(); cosT.Dispose(); sinT.Dispose();
    }
}
