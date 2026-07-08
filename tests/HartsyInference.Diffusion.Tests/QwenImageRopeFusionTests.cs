using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Proves the fused <see cref="QwenImageRope.ApplyJoint"/> (used by the GPU-resident QwenImageBlock) is
/// bit-identical to the original per-stream path: <c>ApplyText(txt)</c> + <c>ApplyImage(img)</c> applied BEFORE the
/// <c>[txt, img]</c> concatenation. RoPE is per-row independent, so roping the concatenated sequence with a combined
/// [txt;img] position table must equal roping each stream then concatenating. This isolates the one Qwen-specific new
/// op in the GPU-residency rewrite from the pre-existing model wiring.</summary>
public unsafe class QwenImageRopeFusionTests
{
    private readonly ITestOutputHelper _output;
    public QwenImageRopeFusionTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ApplyJoint_Equals_PerStream_ThenConcat()
    {
        const int batch = 1, heads = 24, headDim = 128, hPacked = 8, wPacked = 8, txtSeq = 7;
        int imgSeq = hPacked * wPacked;
        int total = txtSeq + imgSeq;
        int posStart = QwenImageRope.ComputeTextPositionStart(hPacked, wPacked);
        QwenImageRope rope = new();
        Assert.Equal(headDim, rope.HeadDim);

        // Deterministic pseudo-random fill.
        static void Fill(Tensor t, int seed)
        {
            Span<float> s = t.AsSpan<float>();
            uint x = (uint)(seed * 2654435761u + 1);
            for (int i = 0; i < s.Length; i++) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; s[i] = ((x >> 8) / (float)(1 << 24)) * 2f - 1f; }
        }

        TensorShape imgShape = new(batch, heads, imgSeq, headDim);
        TensorShape txtShape = new(batch, heads, txtSeq, headDim);
        TensorShape jointShape = new(batch, heads, total, headDim);

        // ── Path A: per-stream rope, then concat [txt, img] along the seq dim ──
        Tensor imgQ = new(imgShape, DType.F32); Fill(imgQ, 1);
        Tensor imgK = new(imgShape, DType.F32); Fill(imgK, 2);
        Tensor txtQ = new(txtShape, DType.F32); Fill(txtQ, 3);
        Tensor txtK = new(txtShape, DType.F32); Fill(txtK, 4);
        rope.ApplyImage(imgQ, imgK, batch, heads, hPacked, wPacked);
        rope.ApplyText(txtQ, txtK, batch, heads, txtSeq, posStart);
        Tensor jointQ_A = ConcatSeq(txtQ, imgQ, batch, heads, txtSeq, imgSeq, headDim);
        Tensor jointK_A = ConcatSeq(txtK, imgK, batch, heads, txtSeq, imgSeq, headDim);

        // ── Path B: concat raw [txt, img], then ApplyJoint ──
        Tensor imgQ2 = new(imgShape, DType.F32); Fill(imgQ2, 1);
        Tensor imgK2 = new(imgShape, DType.F32); Fill(imgK2, 2);
        Tensor txtQ2 = new(txtShape, DType.F32); Fill(txtQ2, 3);
        Tensor txtK2 = new(txtShape, DType.F32); Fill(txtK2, 4);
        Tensor jointQ_B = ConcatSeq(txtQ2, imgQ2, batch, heads, txtSeq, imgSeq, headDim);
        Tensor jointK_B = ConcatSeq(txtK2, imgK2, batch, heads, txtSeq, imgSeq, headDim);
        rope.ApplyJoint(jointQ_B, jointK_B, batch, heads, hPacked, wPacked, txtSeq, posStart);

        float maxQ = MaxAbsDiff(jointQ_A, jointQ_B);
        float maxK = MaxAbsDiff(jointK_A, jointK_B);
        _output.WriteLine($"maxAbsDiff Q={maxQ:E3}  K={maxK:E3} (posStart={posStart}, total={total})");
        Assert.True(maxQ < 1e-6f, $"Q rope fusion diverged: {maxQ}");
        Assert.True(maxK < 1e-6f, $"K rope fusion diverged: {maxK}");
    }

    /// <summary>Proves the DEVICE rope path (cached <see cref="QwenImageRope.GetOrBuildJointTables"/> tables +
    /// <c>IBackend.WanRopeInterleaved</c> on the pre-permute [B, S, H, D] layout, as QwenImageBlock now runs it)
    /// equals the host <see cref="QwenImageRope.ApplyJoint"/> on the post-permute layout. Uses the CpuBackend's
    /// WanRopeInterleaved default implementation, so no GPU is needed.</summary>
    [Fact]
    public void DeviceRopeTables_Equal_HostApplyJoint()
    {
        const int batch = 1, heads = 4, headDim = 128, hPacked = 6, wPacked = 5, txtSeq = 7;
        int imgSeq = hPacked * wPacked;
        int total = txtSeq + imgSeq;
        int posStart = QwenImageRope.ComputeTextPositionStart(hPacked, wPacked);
        QwenImageRope rope = new();

        static void Fill(Tensor t, int seed)
        {
            Span<float> s = t.AsSpan<float>();
            uint x = (uint)(seed * 2654435761u + 1);
            for (int i = 0; i < s.Length; i++) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; s[i] = ((x >> 8) / (float)(1 << 24)) * 2f - 1f; }
        }

        // Device-path layout: pre-permute [B, S, H, D] (flat [S, H·D] rows).
        TensorShape preShape = new(batch, total, heads, headDim);
        Tensor qPre = new(preShape, DType.F32); Fill(qPre, 11);
        Tensor kPre = new(preShape, DType.F32); Fill(kPre, 12);
        (Tensor cos, Tensor sin) = rope.GetOrBuildJointTables(hPacked, wPacked, txtSeq, posStart);
        using HartsyInference.Cpu.CpuBackend cpuBackend = new();
        HartsyInference.Core.Backends.IBackend cpu = cpuBackend;   // WanRopeInterleaved is a default interface method
        cpu.WanRopeInterleaved(qPre, cos, sin, total, heads, headDim);
        cpu.WanRopeInterleaved(kPre, cos, sin, total, heads, headDim);

        // Host-path layout: post-permute [B, H, S, D] with the same starting values.
        TensorShape postShape = new(batch, heads, total, headDim);
        Tensor qPost = new(postShape, DType.F32);
        Tensor kPost = new(postShape, DType.F32);
        Tensor qSrc = new(preShape, DType.F32); Fill(qSrc, 11);
        Tensor kSrc = new(preShape, DType.F32); Fill(kSrc, 12);
        PermuteToHeads(qSrc, qPost, total, heads, headDim);
        PermuteToHeads(kSrc, kPost, total, heads, headDim);
        rope.ApplyJoint(qPost, kPost, batch, heads, hPacked, wPacked, txtSeq, posStart);

        // Compare: permute the device result into the host layout.
        Tensor qDev = new(postShape, DType.F32);
        Tensor kDev = new(postShape, DType.F32);
        PermuteToHeads(qPre, qDev, total, heads, headDim);
        PermuteToHeads(kPre, kDev, total, heads, headDim);
        float maxQ = MaxAbsDiff(qDev, qPost);
        float maxK = MaxAbsDiff(kDev, kPost);
        _output.WriteLine($"device-vs-host rope maxAbsDiff Q={maxQ:E3}  K={maxK:E3}");
        Assert.True(maxQ < 1e-6f, $"Q device rope diverged: {maxQ}");
        Assert.True(maxK < 1e-6f, $"K device rope diverged: {maxK}");
    }

    /// <summary>[B, S, H, D] → [B, H, S, D] host permute (B=1).</summary>
    private static void PermuteToHeads(Tensor src, Tensor dst, int seq, int heads, int headDim)
    {
        float* sp = (float*)src.DataPointer;
        float* dp = (float*)dst.DataPointer;
        for (int s = 0; s < seq; s++)
            for (int h = 0; h < heads; h++)
                for (int d = 0; d < headDim; d++)
                    dp[((long)h * seq + s) * headDim + d] = sp[((long)s * heads + h) * headDim + d];
    }

    /// <summary>Concatenate [txt, img] along the seq dim of [B, H, S, D] (matches the block's joint order).</summary>
    private static Tensor ConcatSeq(Tensor txt, Tensor img, int b, int h, int txtSeq, int imgSeq, int d)
    {
        int total = txtSeq + imgSeq;
        Tensor outT = new(new TensorShape(b, h, total, d), DType.F32);
        float* tp = (float*)txt.DataPointer, ip = (float*)img.DataPointer, op = (float*)outT.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int hi = 0; hi < h; hi++)
            {
                long outBase = ((long)bi * h + hi) * total * d;
                long txtBase = ((long)bi * h + hi) * txtSeq * d;
                long imgBase = ((long)bi * h + hi) * imgSeq * d;
                for (int s = 0; s < txtSeq * d; s++) op[outBase + s] = tp[txtBase + s];
                for (int s = 0; s < imgSeq * d; s++) op[outBase + txtSeq * d + s] = ip[imgBase + s];
            }
        return outT;
    }

    private static float MaxAbsDiff(Tensor a, Tensor b)
    {
        ReadOnlySpan<float> x = a.AsReadOnlySpan<float>(), y = b.AsReadOnlySpan<float>();
        float m = 0; for (int i = 0; i < x.Length; i++) { float d = MathF.Abs(x[i] - y[i]); if (d > m) m = d; }
        return m;
    }
}
