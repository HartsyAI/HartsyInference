using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>SANA-WM GLUMBConvTemp against the upstream module run on CPU in float32
/// (<c>Fixtures/generate_sana_wm_s4_fixtures.py</c>, T=3 and T=1).</summary>
public sealed class SanaWmGlumbTempFfnTests
{
    private const string FixtureT3 = "sana_wm_s4_glumb_temp_ffn_t3.safetensors";
    private const string FixtureT1 = "sana_wm_s4_glumb_temp_ffn_t1.safetensors";

    /// <summary>F32 path against the upstream output.</summary>
    private const float F32Tolerance = 1e-4f;

    /// <summary>BF16 has 8 significant bits, so one rounding moves a value by at most 2^-9 of its magnitude. The operand
    /// rounding here touches the weights and the input only, so the output error stays within 2^-6 of the output scale.</summary>
    private const float Bf16OperandTolerance = 1f / 64;

    private readonly ITestOutputHelper _output;

    public SanaWmGlumbTempFfnTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(FixtureT3)]
    [InlineData(FixtureT1)]
    public void Forward_F32_MatchesUpstreamGlumbConvTemp(string fixture)
    {
        IBackend backend = new CpuBackend();
        using SanaWmGlumbTempCase c = SanaWmGlumbTempCase.Load(fixture, SanaWmOperandMode.Exact);
        using Tensor output = c.Ffn.Forward(backend, c.Input, c.Frames, c.Height, c.Width);

        Assert.Equal(c.Expected.Shape, output.Shape);
        float maxError = MaxAbsDifference(output.AsReadOnlySpan<float>(), c.Expected.AsReadOnlySpan<float>());
        _output.WriteLine($"{fixture}: F32 max_abs_error={maxError:E3} (tolerance {F32Tolerance:E0})");
        Assert.True(maxError <= F32Tolerance, $"F32 max abs error {maxError} exceeds {F32Tolerance}.");
    }

    [Theory]
    [InlineData(FixtureT3)]
    [InlineData(FixtureT1)]
    public void Forward_Bf16RoundedOperands_StaysWithinBf16Tolerance(string fixture)
    {
        IBackend backend = new CpuBackend();
        using SanaWmGlumbTempCase c = SanaWmGlumbTempCase.Load(fixture, SanaWmOperandMode.Bf16Rounded);
        using Tensor output = c.Ffn.Forward(backend, c.Input, c.Frames, c.Height, c.Width);

        Assert.Equal(c.Expected.Shape, output.Shape);
        float scale = c.ExpectedScale();
        float maxError = MaxAbsDifference(output.AsReadOnlySpan<float>(), c.Expected.AsReadOnlySpan<float>());
        _output.WriteLine($"{fixture}: BF16-rounded operands max_abs_error={maxError:E3}, output scale={scale:E3}, "
            + $"relative={maxError / scale:E3} (tolerance {Bf16OperandTolerance:E3} of scale)");
        Assert.True(maxError <= Bf16OperandTolerance * scale, $"BF16-operand max abs error {maxError} exceeds {Bf16OperandTolerance} x {scale}.");
    }

    [Fact]
    public void Forward_ChunkedOneFrameAtATime_MatchesOneChunk()
    {
        IBackend backend = new CpuBackend();
        using SanaWmGlumbTempCase whole = SanaWmGlumbTempCase.Load(FixtureT3, SanaWmOperandMode.Exact);
        using SanaWmGlumbTempCase chunked = SanaWmGlumbTempCase.Load(FixtureT3, SanaWmOperandMode.Exact, hiddenChunkElements: 1);
        int rows = (int)whole.Input.Shape[0] * whole.Frames;
        Assert.Equal(rows, whole.Ffn.ChunkFrames(rows, whole.Height * whole.Width));
        Assert.Equal(1, chunked.Ffn.ChunkFrames(rows, whole.Height * whole.Width));

        using Tensor expected = whole.Ffn.Forward(backend, whole.Input, whole.Frames, whole.Height, whole.Width);
        using Tensor actual = chunked.Ffn.Forward(backend, chunked.Input, chunked.Frames, chunked.Height, chunked.Width);

        Assert.Equal(expected.AsReadOnlySpan<float>().ToArray(), actual.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void Forward_Bf16Activations_OnCpuBackend_ThrowsNotSupported()
    {
        IBackend backend = new CpuBackend();
        using SanaWmGlumbTempCase c = SanaWmGlumbTempCase.Load(FixtureT1, SanaWmOperandMode.Bf16Storage);

        Assert.Throws<NotSupportedException>(() => c.Ffn.Forward(backend, c.Input, c.Frames, c.Height, c.Width));
    }

    [Fact]
    public void Forward_TokenCountNotEqualToGrid_Throws()
    {
        IBackend backend = new CpuBackend();
        using SanaWmGlumbTempCase c = SanaWmGlumbTempCase.Load(FixtureT3, SanaWmOperandMode.Exact);

        Assert.Throws<ArgumentException>(() => c.Ffn.Forward(backend, c.Input, 3, 4, 6));
    }

    private static float MaxAbsDifference(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        float max = 0f;
        for (int i = 0; i < expected.Length; i++)
            max = MathF.Max(max, MathF.Abs(actual[i] - expected[i]));
        return max;
    }
}
