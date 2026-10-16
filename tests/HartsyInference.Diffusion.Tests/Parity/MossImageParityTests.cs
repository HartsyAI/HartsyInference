using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>The tiny checkpoint and reference outputs come from <c>tests/python-reference/moss_image_numpy_reference.py</c>,
/// a float64 transcription of OpenMOSS's ComfyUI module over F16-rounded weights (re-checked against the module itself
/// by <c>dump_moss_image_reference.py</c> wherever torch is available). Fed through
/// <see cref="MossImageCheckpointConverter"/>, the port must reproduce the full forward, the forward with a joint block
/// skipped (the reference's unconditional branch) and the refined text tokens.</summary>
public sealed unsafe class MossImageParityTests(ITestOutputHelper output)
{
    private const float Tolerance = 2e-4f;
    private const int Height = MossImageFixture.Height, Width = MossImageFixture.Width;

    [Fact]
    public void Config_IsReadFromTheWeights() =>
        Run((transformer, _, _) =>
        {
            MossImageConfig config = transformer.Config;
            Assert.Equal(64, config.HiddenSize);
            Assert.Equal(2, config.NumHeads);
            Assert.Equal(32, config.HeadDim);
            Assert.Equal(3, config.NumLayers);
            Assert.Equal(2, config.NumTextBlocks);
            Assert.Equal(96, config.FfnHidden);
            Assert.Equal(16, config.LatentChannels);
            Assert.Equal(48, config.TextDim);
            Assert.Equal(32, config.FreqDim);
        });

    [Fact]
    public void RefinedText_MatchesReference() =>
        Run((transformer, backend, f) =>
        {
            using MossImageConditioning conditioning = transformer.PrepareConditioning(backend, f["inputs.context"], Height, Width);
            AssertClose(f["text.refined"], conditioning.Text, "refined text");
        });

    [Fact]
    public void FullForward_MatchesReference() =>
        Run((transformer, backend, f) =>
        {
            using MossImageConditioning conditioning = transformer.PrepareConditioning(backend, f["inputs.context"], Height, Width);
            using Tensor velocity = transformer.Forward(backend, f["inputs.latent"], 500f, conditioning);
            AssertClose(f["full.velocity"], velocity, "velocity");
        });

    [Fact]
    public void SkippedLayer_MatchesReference() =>
        Run((transformer, backend, f) =>
        {
            using MossImageConditioning conditioning = transformer.PrepareConditioning(backend, f["inputs.context"], Height, Width);
            using Tensor velocity = transformer.Forward(backend, f["inputs.latent"], 937f, conditioning, [1]);
            AssertClose(f["skip.velocity"], velocity, "skipped-layer velocity");
        });

    [Fact]
    public void Branches_MatchSingleForwards() =>
        Run((transformer, backend, f) =>
        {
            using MossImageConditioning conditioning = transformer.PrepareConditioning(backend, f["inputs.context"], Height, Width);
            (Tensor[] velocities, bool callerOwns) = transformer.ForwardBranches(backend, f["inputs.latent"], 937f,
                [new MossImageBranch(conditioning, []), new MossImageBranch(conditioning, [1])]);
            Assert.True(callerOwns);
            using Tensor full = transformer.Forward(backend, f["inputs.latent"], 937f, conditioning);
            AssertClose(full, velocities[0], "branch 0");
            AssertClose(f["skip.velocity"], velocities[1], "branch 1");
            foreach (Tensor velocity in velocities) velocity.Dispose();
        });

    private static void Run(Action<MossImageTransformer, IBackend, IReadOnlyDictionary<string, Tensor>> body)
    {
        using MossImageFixture fixture = new();
        Assert.False(fixture.Weights.ContainsKey("rope.bands"));
        using CpuBackend backend = new();
        body(fixture.Transformer, backend, fixture.Tensors);
    }

    private void AssertClose(Tensor expected, Tensor actual, string what)
    {
        Assert.Equal(expected.ElementCount, actual.ElementCount);
        ReadOnlySpan<float> e = new((void*)expected.DataPointer, (int)expected.ElementCount);
        ReadOnlySpan<float> a = new((void*)actual.DataPointer, (int)actual.ElementCount);
        float worst = 0f, peak = 0f;
        for (int i = 0; i < e.Length; i++)
        {
            Assert.False(float.IsNaN(a[i]), $"{what}: NaN at {i}");
            worst = MathF.Max(worst, MathF.Abs(e[i] - a[i]));
            peak = MathF.Max(peak, MathF.Abs(e[i]));
        }
        output.WriteLine($"{what}: max |Δ| = {worst:E3} (peak {peak:F3})");
        Assert.True(worst <= Tolerance * MathF.Max(1f, peak), $"{what}: max |Δ| {worst:E3} exceeds {Tolerance:E1}·max(1, {peak:F3}).");
    }
}
