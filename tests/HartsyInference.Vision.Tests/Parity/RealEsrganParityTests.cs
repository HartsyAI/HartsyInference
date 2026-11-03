using System.Runtime.InteropServices;
using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.Vision.Upscale;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests.Parity;

/// <summary>Real-ESRGAN against <c>tests/python-reference/real_esrgan_reference.py</c>, a float64 numpy transcription of
/// BasicSR's RRDBNet and RealESRGANer's pre/post-processing at the official CLI defaults (pre_pad 0, no tiling, reflect mod
/// pad). The tiny cases are seeded 8-feature, 2-block networks committed under <c>TestData/RealEsrgan</c> (x4, and x2 with an
/// odd width so the mod pad and unshuffle are exercised); the real case runs Comfy-Org's RealESRGAN_x4plus safetensors on a
/// fixed 16×12 crop. Each checks the converter's inferred config, the network output against the reference, the x2 input
/// tensor, and the pipeline's bytes.</summary>
public sealed class RealEsrganParityTests
{
    private static readonly string FixtureDir = Path.Combine(AppContext.BaseDirectory, "TestData", "RealEsrgan");
    private readonly ITestOutputHelper _output;

    public RealEsrganParityTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData("tiny_x4", 4)]
    [InlineData("tiny_x2", 2)]
    public void TinyNetwork_MatchesTheBasicSrTranscription(string fixtureCase, int scale)
    {
        RunCase(Path.Combine(FixtureDir, fixtureCase + ".safetensors"), fixtureCase, scale, numBlock: 2,
            maxAbsTolerance: 1e-5, maxByteDiff: 0);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void RealX4Plus_MatchesTheBasicSrTranscription()
    {
        string checkpoint = TestPaths.RealEsrgan.X4PlusSafetensors;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint))
            return;
        // Measured 2026-11-03 on CPU: max |Δ| 1.7e-6, mean 2.5e-7, every byte identical. A byte may still flip at a .5
        // boundary on another SIMD path, so one level is allowed.
        RunCase(checkpoint, "real_x4plus", scale: 4, numBlock: 23, maxAbsTolerance: 1e-5, maxByteDiff: 1);
    }

    private void RunCase(string checkpoint, string fixtureCase, int scale, int numBlock, double maxAbsTolerance, int maxByteDiff)
    {
        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(FixtureDir, fixtureCase + ".json")));
        int width = meta.RootElement.GetProperty("width").GetInt32();
        int height = meta.RootElement.GetProperty("height").GetInt32();
        long[] inShape = Shape(meta.RootElement.GetProperty("net_input_shape"));
        long[] outShape = Shape(meta.RootElement.GetProperty("net_output_shape"));

        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(checkpoint);
        Dictionary<string, Tensor> weights = RealEsrganConverter.Convert(loader.GetAllTensors());
        RealEsrganConfig config = RealEsrganConverter.InferConfig(weights);
        Assert.Equal(scale, config.Scale);
        Assert.Equal(numBlock, config.NumBlock);
        RrdbNet net = new RrdbNet(config);
        net.LoadWeights(weights);
        using CpuBackend backend = new CpuBackend();

        byte[] rgb = File.ReadAllBytes(Path.Combine(FixtureDir, fixtureCase + ".input.rgb"));
        Assert.Equal(width * height * 3, rgb.Length);
        float[] refInput = ReadF32(fixtureCase + ".net_input.f32", inShape);
        float[] refOutput = ReadF32(fixtureCase + ".net_output.f32", outShape);

        // The engine's own input tensor (mod pad + unshuffle for x2) must equal the reference's exactly.
        if (scale == 2)
        {
            int padW = width % 2, padH = height % 2;
            using Tensor engineInput = UpscalePipeline.UnshuffledInput(rgb, width, height, 2, padW, padH);
            Assert.Equal(inShape, Dims(engineInput));
            Assert.Equal(refInput, engineInput.AsSpan<float>().ToArray());
        }

        using Tensor input = new Tensor(new TensorShape(inShape), DType.F32);
        refInput.CopyTo(input.AsSpan<float>());
        using Tensor output = net.Forward(backend, input);
        Assert.Equal(outShape, Dims(output));
        (double maxAbs, double meanAbs) = Diff(output.AsSpan<float>(), refOutput);
        _output.WriteLine($"{fixtureCase}: network max |Δ| {maxAbs:E3}, mean |Δ| {meanAbs:E3} (tolerance {maxAbsTolerance:E1})");
        Assert.True(maxAbs <= maxAbsTolerance, $"{fixtureCase}: network max |Δ| {maxAbs:E3} exceeds {maxAbsTolerance:E1}.");

        // Whole pipeline, untiled as the official CLI runs by default: bytes in, bytes out.
        UpscalePipeline pipeline = new UpscalePipeline(backend, net, inputTileSize: 0);
        (byte[] bytes, int outW, int outH) = pipeline.Upscale(rgb, width, height);
        byte[] refBytes = File.ReadAllBytes(Path.Combine(FixtureDir, fixtureCase + ".output.rgb"));
        Assert.Equal((width * scale, height * scale), (outW, outH));
        Assert.Equal(refBytes.Length, bytes.Length);
        int worst = 0, differing = 0;
        for (int i = 0; i < bytes.Length; i++)
        {
            int d = Math.Abs(bytes[i] - refBytes[i]);
            worst = Math.Max(worst, d);
            if (d != 0)
                differing++;
        }
        _output.WriteLine($"{fixtureCase}: pipeline bytes differ at {differing}/{bytes.Length}, worst {worst}");
        Assert.True(worst <= maxByteDiff, $"{fixtureCase}: a pipeline byte differs by {worst} (allowed {maxByteDiff}).");
    }

    private static float[] ReadF32(string file, long[] shape)
    {
        byte[] raw = File.ReadAllBytes(Path.Combine(FixtureDir, file));
        long count = shape.Aggregate(1L, (a, d) => a * d);
        Assert.Equal(count * sizeof(float), raw.Length);
        return MemoryMarshal.Cast<byte, float>(raw).ToArray();
    }

    private static long[] Shape(JsonElement e) => e.EnumerateArray().Select(d => d.GetInt64()).ToArray();

    private static long[] Dims(Tensor t) => Enumerable.Range(0, t.Shape.Rank).Select(i => t.Shape[i]).ToArray();

    private static (double MaxAbs, double MeanAbs) Diff(ReadOnlySpan<float> actual, float[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double max = 0, sum = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = Math.Abs((double)actual[i] - expected[i]);
            max = Math.Max(max, d);
            sum += d;
        }
        return (max, sum / expected.Length);
    }
}
