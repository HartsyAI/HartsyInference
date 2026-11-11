using System.Diagnostics;
using Xunit.Abstractions;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>Integration lane: the released Paint 2.0 checkpoint against the UNet's key map and shapes. Set <c>HARTSY_PAINT20_UNET</c> to the
/// <c>unet/diffusion_pytorch_model.safetensors</c> of <c>tencent/Hunyuan3D-2</c> <c>hunyuan3d-paint-v2-0</c>.</summary>
[Trait("Category", "Integration")]
public sealed class HunyuanPaintRealWeightsTests(ITestOutputHelper output)
{
    [RealWeightsFact("HARTSY_PAINT20_UNET")]
    public void Paint20Unet_LoadsEveryTensorIntoBothStreamsWithoutAMissingKey()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(Environment.GetEnvironmentVariable("HARTSY_PAINT20_UNET")!);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();

        HunyuanPaintConfig config = HunyuanPaintLoader.Detect(weights);
        using HunyuanPaintUnet unet = new(config, generating: true), reference = new(config, generating: false);
        unet.LoadWeights(weights, "unet");
        reference.LoadWeights(weights, "unet_dual");
        HunyuanPaintPromptTokens tokens = HunyuanPaintPromptTokens.FromWeights(weights, config, "unet");

        Assert.Equal(49, config.CameraClasses);
        Assert.Single(tokens.PerMaterial);
        Assert.Equal(new TensorShape(1, 77, 1024), tokens.Reference.Shape);
    }

    /// <summary>One denoising pass on the real 2.0 checkpoint at <c>HARTSY_PAINT_VIEW_LATENT</c> (default 16, so 128 px) over six views: the reference stream writes
    /// its features, is released, then the generating stream runs with and without them. The streams run one after the other so peak memory is one F32 stream.
    /// Checks the output is finite, has the right shape, depends on the reference features and on the camera class. Times and memory go to the test output.</summary>
    [RealWeightsFact("HARTSY_PAINT20_UNET")]
    public void Paint20Unet_OnePassIsFiniteAndUsesTheReferenceFeatures()
    {
        int latent = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_PAINT_VIEW_LATENT"), out int value) ? value : 16;
        const int Views = 6;
        using SafeTensorsLoader loader = new();
        loader.Load(Environment.GetEnvironmentVariable("HARTSY_PAINT20_UNET")!);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();
        HunyuanPaintConfig config = HunyuanPaintLoader.Detect(weights);
        HunyuanPaintPromptTokens tokens = HunyuanPaintPromptTokens.FromWeights(weights, config, "unet");
        using CpuBackend backend = new();
        Process process = Process.GetCurrentProcess();
        Random random = new(1);
        float Gaussian() => (float)(Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble()));

        Stopwatch watch = Stopwatch.StartNew();
        HunyuanPaintBlockState write = new() { Write = true, Text = [tokens.Reference] };
        try
        {
            using (Tensor reference = new(new TensorShape(1, 4, latent, latent), DType.F32))
            {
                Fill(reference, Gaussian);
                using HunyuanPaintUnet referenceStream = new(config, generating: false);
                referenceStream.LoadWeights(weights, "unet_dual");
                output.WriteLine($"reference stream load: {watch.Elapsed.TotalSeconds:F1} s");
                watch.Restart();
                foreach (Tensor t in referenceStream.Forward(backend, [reference], 0f, [], write))
                    t.Dispose();
                output.WriteLine($"reference stream forward ({latent * 8} px): {watch.Elapsed.TotalSeconds:F1} s, {write.Embeds.Count} feature blocks");
            }
            Assert.NotEmpty(write.Embeds);

            watch.Restart();
            using HunyuanPaintUnet generating = new(config, generating: true);
            generating.LoadWeights(weights, "unet");
            output.WriteLine($"generating stream load: {watch.Elapsed.TotalSeconds:F1} s");
            using Tensor input = new(new TensorShape(Views, config.InChannels, latent, latent), DType.F32);
            Fill(input, Gaussian);
            int[] classes = [0, 1, 2, 3, 4, 5];

            watch.Restart();
            HunyuanPaintBlockState conditioned = new() { Embeds = write.Embeds, ReferenceScale = 1f, Text = tokens.PerMaterial };
            float[] withReference = Run(generating, backend, input, classes, conditioned);
            output.WriteLine($"generating forward with reference: {watch.Elapsed.TotalSeconds:F1} s");
            watch.Restart();
            float[] without = Run(generating, backend, input, classes, new HunyuanPaintBlockState { Embeds = write.Embeds, ReferenceScale = 0f, Text = tokens.PerMaterial });
            output.WriteLine($"generating forward without reference: {watch.Elapsed.TotalSeconds:F1} s");
            float[] otherClasses = Run(generating, backend, input, [6, 7, 8, 9, 10, 11], conditioned);
            process.Refresh();
            output.WriteLine($"peak working set: {process.PeakWorkingSet64 / (1024.0 * 1024 * 1024):F2} GiB");

            Assert.Equal(Views * 4 * latent * latent, withReference.Length);
            Assert.All(withReference, v => Assert.True(float.IsFinite(v)));
            double rms = Math.Sqrt(withReference.Sum(v => (double)v * v) / withReference.Length);
            output.WriteLine($"prediction rms: {rms:F4}");
            Assert.InRange(rms, 0.05, 20);
            Assert.True(MaxDifference(withReference, without) > 1e-3, "the reference features did not change the prediction");
            Assert.True(MaxDifference(withReference, otherClasses) > 1e-3, "the camera class did not change the prediction");
        }
        finally
        {
            foreach (Tensor t in write.Embeds.Values)
                t.Dispose();
        }
    }

    private static float[] Run(HunyuanPaintUnet unet, CpuBackend backend, Tensor input, int[] classes, HunyuanPaintBlockState state)
    {
        Tensor[] outputs = unet.Forward(backend, [input], 999f, classes, state);
        try
        {
            return HunyuanPaintImages.ToArray(outputs[0]);
        }
        finally
        {
            foreach (Tensor t in outputs)
                t.Dispose();
        }
    }

    private static unsafe void Fill(Tensor tensor, Func<float> next)
    {
        float* data = (float*)tensor.DataPointer;
        for (long i = 0; i < tensor.ElementCount; i++)
            data[i] = next();
    }

    private static float MaxDifference(float[] a, float[] b)
    {
        float max = 0;
        for (int i = 0; i < a.Length; i++)
            max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }
}
