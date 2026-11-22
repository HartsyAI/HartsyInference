using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>FLUX.2's F16 block loop on Vulkan with weights that are not fp8. The loop keeps its residual stream inside
/// F16's range by damping it 1/32, and the damp rides on each weight's <see cref="Tensor.Fp8ScaleFactor"/>; a backend
/// that drops the factor runs the stream at its natural scale, overflows, and the image comes out black. The tokens
/// here are large enough that the undamped stream passes 65504 while the damped one stays under it.</summary>
public sealed class Flux2F16DampVulkanTests(ITestOutputHelper output)
{
    private const int GridHeight = 2;
    private const int GridWidth = 3;
    private const int TextLength = 5;
    /// <summary>Tokens of about 1e6 give an embedded stream of about 2e5 undamped and 6e3 damped.</summary>
    private const float TokenScale = 5e6f;

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void OnVulkanTheF16LoopStaysInRangeAndMatchesTheF32Loop()
    {
        VulkanBackend backend;
        try { backend = new VulkanBackend(); }
        catch (Exception ex) { output.WriteLine($"SKIPPED: Vulkan unavailable ({ex.Message})"); return; }
        using (backend)
        {
            if (!backend.Capabilities.SupportsF16) { output.WriteLine("SKIPPED: this Vulkan device has no F16"); return; }
            using Tensor tokens = IrisZTestData.Random(new Random(21), 1, GridHeight * GridWidth, IrisZTestData.Channels);
            foreach (ref float value in tokens.AsSpan<float>()) value *= TokenScale;
            using Tensor text = IrisZTestData.Random(new Random(22), 1, TextLength, IrisZTestData.Context);

            float[] f32 = RunLoaded(backend, tokens, text, ditF16: false);
            float[] f16 = RunLoaded(backend, tokens, text, ditF16: true);

            Assert.All(f32, value => Assert.True(float.IsFinite(value), "the F32 loop produced a non-finite velocity"));
            Assert.All(f16, value => Assert.True(float.IsFinite(value), "the F16 loop produced a non-finite velocity"));
            float difference = 0f, magnitude = 0f;
            for (int i = 0; i < f32.Length; i++)
            {
                difference = MathF.Max(difference, MathF.Abs(f32[i] - f16[i]));
                magnitude = MathF.Max(magnitude, MathF.Abs(f32[i]));
            }
            // The final LayerNorm cancels the damp, so what is left is the F16 stream's 11 bits.
            Assert.True(difference <= 0.02f * magnitude, $"max abs difference {difference} against a largest velocity of {magnitude}");
        }
    }

    /// <summary>A fresh checkpoint per load: loading under <c>numerics.ditF16</c> multiplies the damp into the weights' factors.</summary>
    private static float[] RunLoaded(VulkanBackend backend, Tensor tokens, Tensor text, bool ditF16)
    {
        using IDisposable knob = KnobProfile.Create("flux2-f16-damp-test").With(EngineKnobs.DitF16, ditF16).Push();
        Dictionary<string, Tensor> weights = IrisZTestData.Converted(zeroModality: true);
        using Flux2Transformer transformer = new(IrisZTestData.Config(modalities: 0));
        transformer.LoadWeights(weights);
        try
        {
            using Tensor velocity = transformer.Forward(backend, tokens, text, 0.6f, 0f, GridHeight, GridWidth);
            return velocity.AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            backend.FreeWeights(transformer.EnumerateWeights());
        }
    }
}
