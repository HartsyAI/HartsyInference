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
