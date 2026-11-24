using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.DepthPro2;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests.DepthPro2.Parity;

/// <summary>
/// <see cref="DepthPro2Model"/> end to end on CPU F32 against upstream <c>infer</c>: the saved uint8 RGB through <see cref="DepthPro2Preprocessor"/>, the network,
/// the focal estimator and the metric step, compared with the dump of <c>tests/python-reference/dump_depth_pro2.py</c> (which stores infinite depth as 0).
/// Needs <c>HARTSY_DEPTHPRO2_WEIGHTS</c>, <c>HARTSY_DEPTHPRO2_FOCAL_WEIGHTS</c> and <c>HARTSY_DEPTHPRO2_REF</c>; otherwise reported as skipped.
/// </summary>
[Trait("Category", "Integration")]
public sealed class DepthPro2ModelParityTests(ITestOutputHelper output)
{
    private const string WeightsVariable = "HARTSY_DEPTHPRO2_WEIGHTS";
    private const string FocalVariable = "HARTSY_DEPTHPRO2_FOCAL_WEIGHTS";
    private const string ReferenceVariable = "HARTSY_DEPTHPRO2_REF";
    private const double PixelTolerance = 1e-6;
    private const double DepthTolerance = 1e-4;
    private const double FocalTolerance = 1e-4;

    [RealWeightsFact(ReferenceVariable)]
    public void Normalize_MatchesToNormalizedTensor()
    {
        Reference reference = Reference.Open();
        foreach (string name in new[] { "a", "b", "c" })
        {
            (byte[] rgb, int height, int width) = reference.Rgb(name);
            using Tensor pixels = DepthPro2Preprocessor.Normalize(rgb, width, height);
            float[] expected = reference.Read($"{name}_pixels", out _);
            double worst = MaxAbs(pixels.AsReadOnlySpan<float>(), expected);
            output.WriteLine($"{name} pixels max abs {worst:E2}");
            Assert.True(worst <= PixelTolerance, $"{name}: pixels differ by {worst:E2}");
        }
    }

    [RealWeightsFact(WeightsVariable, FocalVariable, ReferenceVariable)]
    public void MetricDepth_WithEstimatedFocal_MatchesInfer_256x384() => RunMetric("a");

    [RealWeightsFact(WeightsVariable, FocalVariable, ReferenceVariable)]
    public void MetricDepth_WithEstimatedFocal_MatchesInfer_Padded250x370() => RunMetric("b");

    [RealWeightsFact(WeightsVariable, ReferenceVariable)]
    public void MetricDepth_WithGivenFocal_UsesItUnchanged()
    {
        Reference reference = Reference.Open();
        (byte[] rgb, int height, int width) = reference.Rgb("a");
        double focal = reference.Read("a_f_px", out _)[0];
        using SafeTensorsLoader loader = Load(WeightsVariable);
        DepthPro2Model model = new(new DepthPro2Network(DepthPro2Preset.Small, loader.GetAllTensors()));
        using CpuBackend backend = new();

        DepthPro2Prediction prediction = model.Predict(backend, [rgb], width, height, focal, focalEstimator: null, keepInverseDepth: false);

        Assert.Null(prediction.Maps[0].InverseDepth);
        Assert.False(prediction.FocalEstimated);
        Assert.Equal(focal, prediction.FocalLengthPx);
        CompareDepth("a (given f_px)", prediction.Maps[0], reference.Read("a_depth", out _));
    }

    private void RunMetric(string name)
    {
        Reference reference = Reference.Open();
        (byte[] rgb, int height, int width) = reference.Rgb(name);
        using SafeTensorsLoader depthLoader = Load(WeightsVariable);
        using SafeTensorsLoader focalLoader = Load(FocalVariable);
        DepthPro2Model model = new(new DepthPro2Network(DepthPro2Preset.Small, depthLoader.GetAllTensors()));
        DepthPro2FocalEstimator estimator = new(focalLoader.GetAllTensors());
        using CpuBackend backend = new();

        DepthPro2Prediction prediction = model.Predict(backend, [rgb], width, height, focalLengthPx: null, estimator);

        double refFocal = reference.Read($"{name}_f_px", out _)[0];
        double focalError = Math.Abs(prediction.FocalLengthPx - refFocal) / refFocal;
        output.WriteLine($"{name} f_px {prediction.FocalLengthPx:F4} vs {refFocal:F4} (rel {focalError:E2})");
        Assert.True(prediction.FocalEstimated);
        Assert.True(focalError <= FocalTolerance, $"{name}: f_px rel {focalError:E2}");
        CompareDepth(name, prediction.Maps[0], reference.Read($"{name}_depth", out _));
    }

    private void CompareDepth(string label, DepthPro2Map map, float[] expected)
    {
        Assert.Equal(expected.Length, map.DepthMeters.Length);
        double diff2 = 0, ref2 = 0;
        int invalid = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            bool referenceInfinite = expected[i] == 0f;
            Assert.Equal(referenceInfinite, float.IsPositiveInfinity(map.DepthMeters[i]));
            Assert.Equal(referenceInfinite ? 0f : 1f, map.ValidMask[i]);
            if (referenceInfinite)
            {
                invalid++;
                continue;
            }

            double d = map.DepthMeters[i] - (double)expected[i];
            diff2 += d * d;
            ref2 += (double)expected[i] * expected[i];
        }

        double relRms = Math.Sqrt(diff2 / Math.Max(ref2, 1e-30));
        output.WriteLine($"{label} depth relRMS {relRms:E2}, {invalid} infinite pixels");
        Assert.True(relRms <= DepthTolerance, $"{label}: depth relRMS {relRms:E2}");
    }

    private static SafeTensorsLoader Load(string variable)
    {
        SafeTensorsLoader loader = new();
        loader.Load(Environment.GetEnvironmentVariable(variable)!);
        return loader;
    }

    private static double MaxAbs(ReadOnlySpan<float> actual, float[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]));
        return worst;
    }

    private sealed class Reference
    {
        private readonly string _dir;
        private readonly JsonElement _meta;

        private Reference(string metaPath)
        {
            _dir = Path.GetDirectoryName(metaPath)!;
            _meta = JsonDocument.Parse(File.ReadAllText(metaPath)).RootElement;
        }

        public static Reference Open() => new(Environment.GetEnvironmentVariable(ReferenceVariable)!);

        public (byte[] Rgb, int Height, int Width) Rgb(string name)
        {
            int[] shape = _meta.GetProperty($"{name}_rgb_u8").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            return (File.ReadAllBytes(Path.Combine(_dir, $"{name}_rgb_u8.bin")), shape[0], shape[1]);
        }

        public float[] Read(string key, out long[] shape)
        {
            shape = _meta.GetProperty(key).EnumerateArray().Select(e => e.GetInt64()).ToArray();
            byte[] bytes = File.ReadAllBytes(Path.Combine(_dir, key + ".bin"));
            float[] values = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }
    }
}
