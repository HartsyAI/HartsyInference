using Xunit;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Gates MiniMax-H3's rotary layout against the op that consumes it. H3 rotates 96 of 128 head dims with
/// NEOX split-half pairing; the nearby <c>Ltx2SplitRope</c> (full rotary) and the interleaved family are both wrong
/// for it, and either mistake produces coherent-but-wrong output rather than a crash.</summary>
public unsafe class MiniMaxH3RopeTests
{
    private const int HeadDim = 128, InvFreqLen = 16;

    [Fact]
    public void RotaryWidthIsThreeAxesDuplicatedAcrossThePairHalves()
    {
        // t/h/w x 16 freqs = 48 angles, duplicated -> 96 of the 128 head dims rotate.
        Assert.Equal(96, MiniMaxH3Rope.RotaryDim(InvFreqLen));
    }

    [Fact]
    public void PairHalvesCarryTheSameAngleAndTheTailStaysZero()
    {
        double[] pos = [3.5, 11.25, 7.0, 1.0, 2.0, 3.0];
        (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables(pos, MiniMaxH3Rope.DefaultInvFreq(InvFreqLen), HeadDim);
        float* c = (float*)cos.DataPointer;
        float* s = (float*)sin.DataPointer;
        int half = 96 / 2;
        for (int row = 0; row < 2; row++)
        {
            long b = (long)row * HeadDim;
            for (int i = 0; i < half; i++)
            {
                Assert.Equal(c[b + i], c[b + i + half], 6);
                Assert.Equal(s[b + i], s[b + i + half], 6);
            }
            // Dims beyond the rotary width are never written and never read.
            for (int i = 96; i < HeadDim; i++)
            {
                Assert.Equal(0f, c[b + i]);
                Assert.Equal(0f, s[b + i]);
            }
        }
    }

    [Fact]
    public void AxesAreLaidOutTimeThenHeightThenWidth()
    {
        // A position that is non-zero on exactly one axis must leave the other axes' angle blocks at cos=1/sin=0.
        float[] inv = MiniMaxH3Rope.DefaultInvFreq(InvFreqLen);
        (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables([0.0, 5.0, 0.0], inv, HeadDim);
        float* c = (float*)cos.DataPointer;
        float* s = (float*)sin.DataPointer;
        for (int i = 0; i < InvFreqLen; i++)
        {
            Assert.Equal(1f, c[i], 6);          // t block: pos 0
            Assert.Equal(0f, s[i], 6);
            Assert.Equal(1f, c[2 * InvFreqLen + i], 6);   // w block: pos 0
            Assert.Equal(0f, s[2 * InvFreqLen + i], 6);
        }
        // h block carries the actual rotation.
        Assert.Equal((float)Math.Cos(5.0 * inv[1]), c[InvFreqLen + 1], 5);
        Assert.Equal((float)Math.Sin(5.0 * inv[1]), s[InvFreqLen + 1], 5);
    }

    [Fact]
    public void ApplyRopeSingleRotatesOnlyTheFirst96DimsAndPreservesPairNorm()
    {
        // ApplyRopeSingle is a default interface method, so it must be called through IBackend.
        IBackend backend = new CpuBackend();
        int heads = 2;
        double[] pos = [2.0, 3.0, 4.0];
        (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables(pos, MiniMaxH3Rope.DefaultInvFreq(InvFreqLen), HeadDim);

        Tensor x = new Tensor(new TensorShape(1, 1, heads, HeadDim), DType.F32);
        float* p = (float*)x.DataPointer;
        for (int i = 0; i < heads * HeadDim; i++) p[i] = (i % 17) * 0.1f + 0.05f;
        float[] before = new float[heads * HeadDim];
        for (int i = 0; i < before.Length; i++) before[i] = p[i];

        backend.ApplyRopeSingle(x, cos, sin, MiniMaxH3Rope.RotaryDim(InvFreqLen));

        int half = 48;
        for (int h = 0; h < heads; h++)
        {
            int b = h * HeadDim;
            // The unrotated tail must be untouched.
            for (int i = 96; i < HeadDim; i++) Assert.Equal(before[b + i], p[b + i], 6);
            // A 2-D rotation preserves the norm of each (i, i+half) pair.
            for (int i = 0; i < half; i++)
            {
                double n0 = before[b + i] * before[b + i] + before[b + i + half] * before[b + i + half];
                double n1 = (double)p[b + i] * p[b + i] + (double)p[b + i + half] * p[b + i + half];
                Assert.Equal(n0, n1, 4);
            }
            // And it must actually have rotated something.
            Assert.NotEqual(before[b], p[b], 6);
        }
    }

    [Fact]
    public void RotaryWiderThanTheHeadIsRejectedRatherThanSilentlyTruncated()
    {
        Assert.Throws<ArgumentException>(() =>
            MiniMaxH3Rope.BuildTables([0.0, 0.0, 0.0], MiniMaxH3Rope.DefaultInvFreq(32), headDim: 64));
    }
}
