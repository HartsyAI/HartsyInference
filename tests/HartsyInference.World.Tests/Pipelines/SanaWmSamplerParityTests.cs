using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Codec;
using HartsyInference.World.Pipelines;
using HartsyInference.World.Tests.Camera;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.World.Tests.Pipelines;

/// <summary>The pipeline's host arithmetic against run <c>A_demo_pose</c>: the guidance combine and Euler update of step 0
/// replayed from the recorded DiT outputs, and the seed-image preprocessing. CPU only; needs <c>SANAWM_REFERENCE_RUNS_DIR</c>.</summary>
[Trait("Category", "Integration")]
public sealed class SanaWmSamplerParityTests
{
    private readonly ITestOutputHelper _output;

    public SanaWmSamplerParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [SanaWmReferenceRunsFact]
    public void StepZero_ReplayedFromTheRecordedDitOutputs_ReproducesTheRecordedLatent()
    {
        string run = SanaWmReferenceRunsFactAttribute.Run("A_demo_pose");
        using SafeTensorsLoader sampler = Load(Path.Combine(run, "sampler.safetensors"));
        float[] input = Floats(sampler.GetTensor("dit_in_latents_step0"));
        float[] output = Floats(sampler.GetTensor("dit_out_raw_step0"));
        float[] timesteps = Floats(sampler.GetTensor("dit_timestep_step0"));
        float[] expected = Floats(sampler.GetTensor("latent_after_step0"));
        int half = input.Length / 2;
        float[] latents = input[..half];
        float[] guided = output[..half];
        float[] conditionFrame = FrameZero(latents);

        SanaWmPipeline.CombineGuidance(guided, output[half..], 5.0);
        (float[] sigmas, _) = SanaWmSchedule.Build(60, 9.8);
        SanaWmPipeline.EulerStep(latents, guided, timesteps[1], sigmas, conditionFrame);

        int mismatched = 0;
        float maxError = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            float error = MathF.Abs(expected[i] - latents[i]);
            maxError = MathF.Max(maxError, error);
            if (error != 0f)
                mismatched++;
        }
        _output.WriteLine($"latent_after_step0: {mismatched} of {expected.Length} values differ, max abs error {maxError:E3}");
        Assert.Equal(0f, timesteps[0]);
        Assert.Equal(0, mismatched);
    }

    [SanaWmReferenceRunsFact]
    public void SeedImagePreprocessing_MatchesThePilLanczosReference()
    {
        string run = SanaWmReferenceRunsFactAttribute.Run("A_demo_pose");
        using System.Text.Json.JsonDocument index = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(Path.Combine(run, "index.json")));
        string imagePath = index.RootElement.GetProperty("args").GetProperty("image").GetString()!;
        (byte[] rgb, int width, int height) = PngDecoder.DecodeFromFile(imagePath);
        using SafeTensorsLoader image = Load(Path.Combine(run, "image.safetensors"));
        float[] expected = Floats(image.GetTensor("image_fp32"));

        using Tensor actual = SanaWmPipeline.PrepareImage(rgb, width, height);

        ReadOnlySpan<float> values = actual.AsReadOnlySpan<float>();
        Assert.Equal(expected.Length, values.Length);
        int differing = 0;
        float maxLevels = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            float levels = MathF.Abs(expected[i] - values[i]) * 127.5f;
            maxLevels = MathF.Max(maxLevels, levels);
            if (levels > 0.5f)
                differing++;
        }
        _output.WriteLine($"image_fp32: {differing} of {expected.Length} values differ, max {maxLevels:F2} 8-bit levels");
        Assert.Equal(0, differing);
    }

    private static float[] FrameZero(float[] latents)
    {
        const int Channels = 128, Frames = 21, Plane = 22 * 40;
        float[] frame = new float[Channels * Plane];
        for (int c = 0; c < Channels; c++)
            Array.Copy(latents, c * Frames * Plane, frame, c * Plane, Plane);
        return frame;
    }

    private static SafeTensorsLoader Load(string path)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        return loader;
    }

    private static float[] Floats(Tensor tensor)
    {
        if (tensor.DType == DType.F32)
            return tensor.AsReadOnlySpan<float>().ToArray();
        using Tensor converted = tensor.CastTo(DType.F32);
        return converted.AsReadOnlySpan<float>().ToArray();
    }
}
