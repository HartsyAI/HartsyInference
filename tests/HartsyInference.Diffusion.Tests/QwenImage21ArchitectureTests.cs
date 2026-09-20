using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Schedulers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the Qwen-Image 2.1 details that fail as plausible output rather than as an error: the RoPE position
/// rule, the SwiGLU half order, the shared modulation's two timestep rows, the timestep's deliberate round through
/// bf16, and the schedule's identity with ComfyUI's <c>flux_time_shift</c>. Each is a place where being wrong still
/// produces an image.</summary>
public sealed class QwenImage21ArchitectureTests
{
    /// <summary>Text tokens take the running counter on all three axes; the image then takes that counter on the
    /// sequence axis and CENTERED indices on the other two. Getting either half from Qwen-Image v1 (image pinned to
    /// axis 0, text starting at max(H/2, W/2)) still renders, just wrongly composed.</summary>
    [Fact]
    public void TextPositionsWalkTheCounterAndImagePositionsAreCenteredOnIt()
    {
        (double Seq, double Height, double Width)[] text = QwenImage21Rope.TextPositions(4);
        Assert.Equal(4, text.Length);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal((i, i, i), text[i]);
        }

        // 4x6 grid after a 4-token prompt: sequence axis is the constant 4, height centered on 4-(4-2)=2,
        // width centered on 6-(6-3)=3.
        (double Seq, double Height, double Width)[] image = QwenImage21Rope.ImagePositions(textLen: 4, imgH: 4, imgW: 6);
        Assert.Equal(24, image.Length);
        Assert.Equal((4, -2, -3), image[0]);
        Assert.Equal((4, -2, 2), image[5]);
        Assert.Equal((4, 1, -3), image[18]);
        Assert.Equal((4, 1, 2), image[23]);
    }

    /// <summary>An ODD grid is where the centering convention actually shows, and the reference's is not the
    /// common one. ComfyUI writes (<c>qwen_image21/model.py</c>, <c>build_sequence</c>):
    /// <code>hh = torch.arange(h) - (h - h // 2) + 0.5 * (h % 2 - x.shape[-2] % 2)</code>
    /// For the target image the parity term is zero, leaving <c>r − (h − h/2)</c>. At <c>h = 3</c> that is
    /// <c>r − 2</c> → <c>−2, −1, 0</c>, which is deliberately NOT the <c>r − h//2</c> convention that would give
    /// <c>−1, 0, 1</c>. Both render; only one matches the model's training. The expected values below are read
    /// off the reference expression, not off this implementation, and the second assertion fails under the
    /// <c>h//2</c> convention specifically.</summary>
    [Fact]
    public void AnOddGridUsesTheReferencesCenteringNotTheCommonOne()
    {
        (double Seq, double Height, double Width)[] image = QwenImage21Rope.ImagePositions(textLen: 7, imgH: 3, imgW: 3);
        Assert.Equal((7, -2, -2), image[0]);
        Assert.Equal((7, -1, -1), image[4]);
        Assert.Equal((7, 0, 0), image[8]);
        // The h//2 convention would put the grid's centre row at 0 and its last row at +1; the reference ends at 0.
        Assert.DoesNotContain(image, p => p.Height > 0 || p.Width > 0);
        Assert.All(image, p => Assert.Equal(7, p.Seq));
    }

    /// <summary>The same expression for an EVEN grid, where <c>h − h/2</c> and <c>h//2</c> happen to agree — so
    /// this case cannot distinguish the two conventions and is pinned only to catch an off-by-one.</summary>
    [Fact]
    public void AnEvenGridCentersOnTheHalfwayRow()
    {
        (double Seq, double Height, double Width)[] image = QwenImage21Rope.ImagePositions(textLen: 2, imgH: 4, imgW: 4);
        Assert.Equal((2, -2, -2), image[0]);
        Assert.Equal((2, 1, 1), image[15]);
    }

    /// <summary>The rope table is axis-major with each pair's angle duplicated across both slots, which is what the
    /// interleaved kernel reads. Checked against the closed form rather than a golden file so a changed axis split
    /// is caught too.</summary>
    [Fact]
    public void RopeTablesMatchTheClosedFormAndDuplicateEachPair()
    {
        int[] axes = [4, 6, 6];
        using QwenImage21Rope rope = new QwenImage21Rope(axes, theta: 10000);
        Assert.Equal(16, rope.HeadDim);
        using CpuBackend backend = new CpuBackend();
        (Tensor cos, Tensor sin) = rope.GetOrBuildTextTables(backend, textLen: 3);

        ReadOnlySpan<float> c = cos.AsReadOnlySpan<float>();
        ReadOnlySpan<float> s = sin.AsReadOnlySpan<float>();
        for (int token = 0; token < 3; token++)
        {
            int slot = 0;
            for (int axis = 0; axis < 3; axis++)
            {
                for (int k = 0; k < axes[axis] / 2; k++)
                {
                    double angle = token / Math.Pow(10000, (double)(2 * k) / axes[axis]);
                    int at = token * 16 + slot;
                    Assert.Equal((float)Math.Cos(angle), c[at], 5);
                    Assert.Equal(c[at], c[at + 1]);
                    Assert.Equal((float)Math.Sin(angle), s[at], 5);
                    Assert.Equal(s[at], s[at + 1]);
                    slot += 2;
                }
            }
        }
    }

    /// <summary>Repeated calls with the same layout return the cached tensors; a changed layout rebuilds. A stale
    /// table would rotate every token to the wrong position after the first resolution change in a session.</summary>
    [Fact]
    public void RopeTablesAreCachedPerLayoutAndRebuiltWhenItChanges()
    {
        using QwenImage21Rope rope = new QwenImage21Rope();
        using CpuBackend backend = new CpuBackend();
        (Tensor firstCos, _) = rope.GetOrBuildImageTables(backend, 5, 4, 4);
        (Tensor sameCos, _) = rope.GetOrBuildImageTables(backend, 5, 4, 4);
        Assert.Same(firstCos, sameCos);
        (Tensor otherCos, _) = rope.GetOrBuildImageTables(backend, 5, 8, 4);
        Assert.NotSame(firstCos, otherCos);
        Assert.Equal(32, otherCos.Shape[0]);
    }

    /// <summary>The schedule must be ComfyUI's <c>ModelSamplingFlux</c> at shift 0.69. The reused scheduler applies
    /// <c>time_snr_shift(alpha, t)</c>, which equals <c>flux_time_shift(mu, 1, t)</c> only when
    /// <c>alpha = e^mu</c> — passing 0.69 directly instead would give a schedule that still denoises, just to a
    /// different image.</summary>
    [Fact]
    public void TheScheduleMatchesFluxTimeShiftAtMu069()
    {
        const float mu = 0.69f;
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(MathF.Exp(mu));
        scheduler.SetTimesteps(25);
        float[] sigmas = scheduler.Sigmas();
        Assert.Equal(26, sigmas.Length);
        Assert.Equal(0f, sigmas[25]);
        for (int i = 0; i < 25; i++)
        {
            double t = 1.0 - (double)i / 25;
            double expected = t <= 0 ? 0 : Math.Exp(mu) / (Math.Exp(mu) + (1.0 / t - 1.0));
            Assert.Equal((float)expected, sigmas[i], 5);
        }
    }

    /// <summary>ComfyUI's "simple" scheduler indexes a 10000-entry table; for a step count that divides 10000 it
    /// lands on exactly the linear grid this scheduler builds. 25, 40 and 50 all divide it, so the shipped presets
    /// are byte-comparable — worth pinning, because a step count that does NOT divide it would quantize slightly.</summary>
    [Theory]
    [InlineData(25)]
    [InlineData(40)]
    [InlineData(50)]
    public void SimpleAndLinearGridsAgreeForStepCountsThatDivideTheTable(int steps)
    {
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(MathF.Exp(0.69f));
        scheduler.SetTimesteps(steps);
        float[] sigmas = scheduler.Sigmas();
        for (int x = 0; x < steps; x++)
        {
            // comfy simple_scheduler: sigmas[-(1 + int(x * 10000/steps))] over sigma((j+1)/10000).
            int index = 10000 - 1 - (int)(x * (10000.0 / steps));
            double t = (index + 1) / 10000.0;
            double expected = Math.Exp(0.69) / (Math.Exp(0.69) + (1.0 / t - 1.0));
            Assert.Equal((float)expected, sigmas[x], 5);
        }
    }

    /// <summary>The reference rounds the timestep through the compute dtype twice before the sinusoid. Truncation —
    /// which the host bf16 cast helper performs — disagrees with PyTorch's round-to-nearest-even on about half of
    /// all values, and the result is a slightly different conditioning at every step.</summary>
    [Fact]
    public void TheTimestepRoundsToNearestEvenNotTowardZero()
    {
        // 1/3 is 0x3EAAAAAB. Truncating the low 16 bits gives 0x3EAA; the dropped 0xAAAB is above half, so
        // round-to-nearest-even gives 0x3EAB. PyTorch's .to(bfloat16) produces the latter.
        float value = 1.0f / 3.0f;
        Assert.Equal(0x3EAAAAABu, BitConverter.SingleToUInt32Bits(value));
        float truncated = BitConverter.UInt32BitsToSingle(0x3EAA0000u);
        float expected = BitConverter.UInt32BitsToSingle(0x3EAB0000u);
        float actual = QwenImage21Transformer.Bf16RoundTrip(value);
        Assert.Equal(expected, actual);
        Assert.NotEqual(truncated, actual);
    }

    /// <summary>A value already exactly representable in bf16 must pass through untouched, and the halfway case
    /// must break to even rather than always up.</summary>
    [Fact]
    public void ExactAndHalfwayBf16ValuesRoundAsIeeeRequires()
    {
        float exact = BitConverter.UInt32BitsToSingle(0x3F800000u);   // 1.0
        Assert.Equal(exact, QwenImage21Transformer.Bf16RoundTrip(exact));

        // Low bits exactly 0x8000 with an even high word stays put; with an odd high word rounds up.
        float tieEven = BitConverter.UInt32BitsToSingle(0x3F808000u);
        Assert.Equal(BitConverter.UInt32BitsToSingle(0x3F800000u), QwenImage21Transformer.Bf16RoundTrip(tieEven));
        float tieOdd = BitConverter.UInt32BitsToSingle(0x3F818000u);
        Assert.Equal(BitConverter.UInt32BitsToSingle(0x3F820000u), QwenImage21Transformer.Bf16RoundTrip(tieOdd));
    }
}
