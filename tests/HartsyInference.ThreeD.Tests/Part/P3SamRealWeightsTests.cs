using System.Diagnostics;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Hunyuan3D.Part;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Part;

/// <summary>Integration lane: the real <c>p3sam/p3sam.safetensors</c> of <c>tencent/Hunyuan3D-Part</c> (set <c>HARTSY_P3SAM</c>) against the key map and shapes, and a
/// first forward pass on the CPU over an ellipsoid. Times go to the test output.</summary>
[Trait("Category", "Integration")]
public sealed class P3SamRealWeightsTests(ITestOutputHelper output)
{
    [RealWeightsFact("HARTSY_P3SAM")]
    public void Checkpoint_LoadsEveryTensorAndRunsFeaturesAndHeadsOnTheCpu()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(Environment.GetEnvironmentVariable("HARTSY_P3SAM")!);
        Dictionary<string, HartsyInference.Core.Tensors.Tensor> state = loader.GetAllTensors();
        output.WriteLine($"{state.Count} tensors");
        using CpuBackend backend = new();
        Stopwatch watch = Stopwatch.StartNew();
        using P3SamFeatures features = new(backend, state);
        using P3SamHeads heads = new(state);
        output.WriteLine($"encoder and heads built: {watch.Elapsed.TotalSeconds:F1} s, feature width {features.FeatureDim}");
        Assert.Equal(512, features.FeatureDim);
        Assert.Equal(512, heads.FeatureDim);

        // An ellipsoid of 6000 surface points, normalised to the unit cube.
        Random random = new(1);
        int count = 6000;
        double[] points = new double[count * 3];
        float[] normals = new float[count * 3];
        for (int i = 0; i < count; i++)
        {
            double theta = Math.Acos(1 - 2 * random.NextDouble()), phi = 2 * Math.PI * random.NextDouble();
            double x = Math.Sin(theta) * Math.Cos(phi), y = Math.Cos(theta), z = Math.Sin(theta) * Math.Sin(phi);
            points[i * 3] = 0.6 * x;
            points[i * 3 + 1] = 1.0 * y;
            points[i * 3 + 2] = 0.8 * z;
            double nx = x / 0.6, ny = y / 1.0, nz = z / 0.8, length = Math.Sqrt(nx * nx + ny * ny + nz * nz);
            normals[i * 3] = (float)(nx / length);
            normals[i * 3 + 1] = (float)(ny / length);
            normals[i * 3 + 2] = (float)(nz / length);
        }

        watch.Restart();
        float[] feature = features.Extract(points, normals, count, P3SamVoxelPick.First);
        output.WriteLine($"features for {count} points: {watch.Elapsed.TotalSeconds:F1} s");
        Assert.Equal(count * 512, feature.Length);
        Assert.All(feature, v => Assert.True(float.IsFinite(v)));
        double mean = feature.Average(), std = Math.Sqrt(feature.Average(v => (v - mean) * (v - mean)));
        output.WriteLine($"feature mean {mean:F4} std {std:F4}");

        float[] unit = new float[count * 3];
        for (int i = 0; i < unit.Length; i++)
            unit[i] = (float)points[i];
        watch.Restart();
        P3SamHeads.Prepared prepared = heads.Prepare(backend, feature, unit, count);
        output.WriteLine($"head setup: {watch.Elapsed.TotalSeconds:F1} s");
        watch.Restart();
        P3SamHeads.Result result = heads.Segment(backend, prepared, unit.AsSpan(0, 3));
        output.WriteLine($"one prompt: {watch.Elapsed.TotalSeconds:F1} s; predicted IoU {string.Join(", ", result.Iou.Select(v => v.ToString("F3")))}; mask sizes {string.Join(", ", result.Masks.Select(m => m.Count(p => p > 0.5f)))}");
        Assert.All(result.Iou, v => Assert.InRange(v, 0f, 1f));
        Assert.All(result.Masks.SelectMany(m => m), v => Assert.True(float.IsFinite(v)));
    }

    private static (double MaxAbs, double RelRms, double Correlation) Compare(float[] actual, float[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double max = 0, diff2 = 0, ref2 = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double a = actual[i], b = expected[i], d = a - b;
            max = Math.Max(max, Math.Abs(d));
            diff2 += d * d;
            ref2 += b * b;
            sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
        }

        double n = expected.Length, cov = sab / n - sa / n * (sb / n), va = saa / n - sa / n * (sa / n), vb = sbb / n - sb / n * (sb / n);
        return (max, Math.Sqrt(diff2 / Math.Max(ref2, 1e-30)), cov / Math.Sqrt(Math.Max(va * vb, 1e-30)));
    }

    private static unsafe float[] ToArray(HartsyInference.Core.Tensors.Tensor t)
    {
        float[] values = new float[t.ElementCount];
        new ReadOnlySpan<float>((float*)t.DataPointer, values.Length).CopyTo(values);
        return values;
    }

    private void Report(string stage, float[] actual, float[] expected, double relTolerance)
    {
        (double max, double rel, double corr) = Compare(actual, expected);
        output.WriteLine($"{stage,-26} n={expected.Length,9}  max|diff|={max:E3}  relRMS={rel:E3}  corr={corr:F6}  refMax={expected.Max(Math.Abs):F3}");
        Assert.True(rel < relTolerance && corr > 0.9999, $"{stage}: relRMS {rel:E3}, correlation {corr:F6} are outside the tolerance ({relTolerance:E1}, 0.9999)");
    }

    /// <summary>The Sonata features and the heads against upstream's modules on the CPU (<c>tests/python-reference/dump_p3sam_features.py</c>); set
    /// <c>HARTSY_P3SAM_REF</c> to the fixture next to <c>HARTSY_P3SAM</c>.</summary>
    [RealWeightsFact("HARTSY_P3SAM", "HARTSY_P3SAM_REF")]
    public void FeaturesAndHeads_MatchUpstreamOnTheCpu()
    {
        using SafeTensorsLoader loader = new(), fixtureLoader = new();
        loader.Load(Environment.GetEnvironmentVariable("HARTSY_P3SAM")!);
        fixtureLoader.Load(Environment.GetEnvironmentVariable("HARTSY_P3SAM_REF")!);
        Dictionary<string, HartsyInference.Core.Tensors.Tensor> state = loader.GetAllTensors(), fixture = fixtureLoader.GetAllTensors();
        float[] pointsFloat = ToArray(fixture["points"]);
        int count = pointsFloat.Length / 3;
        double[] points = [.. pointsFloat.Select(v => (double)v)];
        float[] normals = ToArray(fixture["normals"]);
        using CpuBackend backend = new();
        using P3SamFeatures features = new(backend, state);

        Stopwatch watch = Stopwatch.StartNew();
        P3SamFeatures.Detail detail = features.ExtractDetailed(points, normals, count, P3SamVoxelPick.First);
        output.WriteLine($"features for {count} points ({detail.Grid.Selected.Length} voxels): {watch.Elapsed.TotalSeconds:F1} s");

        int[] expectedSelected = [.. ToArray(fixture["selected"]).Select(v => (int)v)], expectedInverse = [.. ToArray(fixture["inverse"]).Select(v => (int)v)];
        int[] expectedGrid = [.. ToArray(fixture["grid_coord"]).Select(v => (int)v)];
        output.WriteLine($"grid sample: selected differ {detail.Grid.Selected.Zip(expectedSelected).Count(p => p.First != p.Second)}, inverse differ {detail.Grid.Inverse.Zip(expectedInverse).Count(p => p.First != p.Second)}, grid coord differ {detail.Grid.GridCoord.Zip(expectedGrid).Count(p => p.First != p.Second)}");
        Assert.Equal(expectedSelected, detail.Grid.Selected);
        Assert.Equal(expectedInverse, detail.Grid.Inverse);
        Assert.Equal(expectedGrid, detail.Grid.GridCoord);

        for (int s = 0; s < detail.Stages.Length; s++)
        {
            int[] expectedStageGrid = [.. ToArray(fixture[$"grid{s}"]).Select(v => (int)v)];
            int differing = detail.StageGrids[s].Length != expectedStageGrid.Length ? -1 : detail.StageGrids[s].Zip(expectedStageGrid).Count(p => p.First != p.Second);
            output.WriteLine($"stage {s} grid coordinates: {detail.StageGrids[s].Length / 3} rows, expected {expectedStageGrid.Length / 3}, differing values {differing}");
            Assert.Equal(expectedStageGrid, detail.StageGrids[s]);
            Report($"sonata stage {s}", detail.Stages[s], ToArray(fixture[$"stage{s}"]), 1e-3);
        }
        Report("gathered 1232 channels", detail.Gathered, ToArray(fixture["gathered"]), 1e-3);
        Report("features (512, per point)", detail.Features, ToArray(fixture["features"]), 1e-3);

        // The heads take the features of the fixture, not the C# ones, so the two halves are checked on their own.
        using P3SamHeads heads = new(state);
        P3SamHeads.Prepared prepared = heads.Prepare(backend, ToArray(fixture["features"]), pointsFloat, count);
        for (int k = 0; k < 2; k++)
        {
            float[] prompt = ToArray(fixture[$"prompt_{k}"]);
            watch.Restart();
            P3SamHeads.Result result = heads.Segment(backend, prepared, prompt);
            output.WriteLine($"prompt {k}: {watch.Elapsed.TotalSeconds:F2} s");
            float[] expectedMask = ToArray(fixture[$"mask_{k}"]); // [count, 3]
            float[] actualMask = new float[expectedMask.Length];
            for (int i = 0; i < count; i++)
                for (int m = 0; m < 3; m++)
                    actualMask[i * 3 + m] = result.Masks[m][i];
            Report($"mask probabilities, prompt {k}", actualMask, expectedMask, 1e-5);
            Report($"predicted IoU, prompt {k}", result.Iou, ToArray(fixture[$"iou_{k}"]), 1e-5);
        }
    }
}
