using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Matting.BiRefNet;

namespace HartsyInference.Vision.Tests.Matting;

/// <summary>Real-weight parity for BiRefNet-matting against the upstream PyTorch model. Needs
/// <c>BIREFNET_MATTING_PATH</c> (the verified <c>model.safetensors</c>) and <c>BIREFNET_MATTING_REF</c> (the
/// <c>reference.safetensors</c> written by <c>tests/python-reference/dump_birefnet_matting.py</c>); reported as skipped
/// when either is absent.</summary>
[Trait("Category", "Integration")]
public sealed class BiRefNetMattingParityTests
{
    private const string WeightsVariable = "BIREFNET_MATTING_PATH";
    private const string ReferenceVariable = "BIREFNET_MATTING_REF";

    [RealWeightsFact(WeightsVariable, ReferenceVariable)]
    public void Matte_MatchesUpstreamPyTorchReference()
    {
        string weightsPath = Environment.GetEnvironmentVariable(WeightsVariable)!;
        string referencePath = Environment.GetEnvironmentVariable(ReferenceVariable)!;

        using SafeTensorsLoader checkpoint = new();
        checkpoint.Load(weightsPath);
        using BiRefNetMatter matter = new();
        matter.LoadWeights(BiRefNetCheckpointConverter.Convert(checkpoint.GetAllTensors()));

        using SafeTensorsLoader reference = new();
        reference.Load(referencePath);
        Tensor inputRgb = reference.GetTensor("input_rgb");
        int height = (int)inputRgb.Shape[0], width = (int)inputRgb.Shape[1];
        byte[] rgb = ToBytes(inputRgb);

        using IBackend backend = new CpuBackend();

        Tensor refPixels = reference.GetTensor("pixels");
        using Tensor ourPixels = BiRefNetMatter.Preprocess(rgb, width, height);
        MaxAndMean(refPixels, ourPixels, out float pixelMax, out float pixelMean);

        Tensor refLogits = reference.GetTensor("logits");
        using Tensor ourLogits = matter.Logits(backend, refPixels);
        MaxAndMean(refLogits, ourLogits, out float logitMax, out float logitMean);
        float correlation = Correlation(refLogits, ourLogits);

        Tensor refMask = reference.GetTensor("mask_u8");
        float[] ourMatte = matter.Matte(backend, rgb, width, height);
        ReadOnlySpan<float> refSpan = refMask.AsReadOnlySpan<float>();
        int offByMore = 0;
        float matteMax = 0f;
        for (int i = 0; i < ourMatte.Length; i++)
        {
            float diff = MathF.Abs(MathF.Round(ourMatte[i] * 255f) - refSpan[i]);
            matteMax = MathF.Max(matteMax, diff);
            if (diff > 1f) offByMore++;
        }

        Assert.True(logitMax < 0.05f, $"logits maxAbs {logitMax} (mean {logitMean}); corr {correlation}");
        Assert.True(correlation > 0.9999f, $"logits correlation {correlation}");
        Assert.True(pixelMax < 0.1f, $"preprocess maxAbs {pixelMax} (mean {pixelMean})");
        Assert.True(matteMax <= 2f && offByMore <= ourMatte.Length / 1000,
            $"matte max diff {matteMax} LSB; {offByMore}/{ourMatte.Length} pixels off by more than 1 LSB");
    }

    private static byte[] ToBytes(Tensor values)
    {
        ReadOnlySpan<float> span = values.AsReadOnlySpan<float>();
        byte[] bytes = new byte[span.Length];
        for (int i = 0; i < span.Length; i++) bytes[i] = (byte)Math.Clamp(MathF.Round(span[i]), 0f, 255f);
        return bytes;
    }

    private static void MaxAndMean(Tensor reference, Tensor actual, out float max, out float mean)
    {
        ReadOnlySpan<float> r = reference.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        max = 0f;
        double sum = 0.0;
        for (int i = 0; i < r.Length; i++)
        {
            float d = MathF.Abs(r[i] - a[i]);
            max = MathF.Max(max, d);
            sum += d;
        }
        mean = (float)(sum / r.Length);
    }

    private static float Correlation(Tensor reference, Tensor actual)
    {
        ReadOnlySpan<float> r = reference.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        double meanR = 0, meanA = 0;
        for (int i = 0; i < r.Length; i++) { meanR += r[i]; meanA += a[i]; }
        meanR /= r.Length; meanA /= r.Length;
        double cov = 0, varR = 0, varA = 0;
        for (int i = 0; i < r.Length; i++)
        {
            double dr = r[i] - meanR, da = a[i] - meanA;
            cov += dr * da; varR += dr * dr; varA += da * da;
        }
        return (float)(cov / Math.Sqrt(varR * varA));
    }
}
