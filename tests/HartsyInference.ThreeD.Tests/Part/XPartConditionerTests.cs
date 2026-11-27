using System.Diagnostics;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Hunyuan3D.Part;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Part;

/// <summary>Integration lane: the X-Part conditioner (object and part encoders, Sonata features, nearest-point gather, output projections) against upstream's modules on the CPU. Set
/// <c>HARTSY_XPART_COND_REF</c> (from <c>tests/python-reference/dump_xpart_conditioner.py</c>) and <c>HARTSY_XPART_CONDITIONER</c> (<c>conditioner.safetensors</c>).</summary>
[Trait("Category", "Integration")]
public sealed class XPartConditionerTests(ITestOutputHelper output)
{
    private static unsafe float[] Values(Tensor t)
    {
        float[] values = new float[t.ElementCount];
        new ReadOnlySpan<float>((float*)t.DataPointer, values.Length).CopyTo(values);
        return values;
    }

    private void Compare(string stage, float[] actual, float[] expected, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        double max = 0, diff2 = 0, ref2 = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = (double)actual[i] - expected[i];
            max = Math.Max(max, Math.Abs(d));
            diff2 += d * d;
            ref2 += (double)expected[i] * expected[i];
        }

        double rel = Math.Sqrt(diff2 / ref2);
        output.WriteLine($"{stage,-24} n={expected.Length,8}  max|diff|={max:E3}  relRMS={rel:E3}");
        Assert.True(rel < tolerance, $"{stage}: relRMS {rel:E3} is above {tolerance:E1}");
    }

    [RealWeightsFact("HARTSY_XPART_COND_REF", "HARTSY_XPART_CONDITIONER")]
    public void Conditioner_MatchesUpstreamOnTheCpu()
    {
        using SafeTensorsLoader refLoader = new(), loader = new();
        refLoader.Load(Environment.GetEnvironmentVariable("HARTSY_XPART_COND_REF")!);
        loader.Load(Environment.GetEnvironmentVariable("HARTSY_XPART_CONDITIONER")!);
        Dictionary<string, Tensor> fixture = refLoader.GetAllTensors(), weights = loader.GetAllTensors();
        float[] points = Values(fixture["points"]), feats = Values(fixture["feats"]);
        int[] objectQueries = Values(fixture["query_idx"]).Select(v => (int)v).ToArray();
        List<XPartPartSample> parts = [];
        for (int k = 0; k < 2; k++)
            parts.Add(new XPartPartSample(Values(fixture[$"part_points.{k}"]), Values(fixture[$"part_feats.{k}"]), Values(fixture[$"part_query_idx.{k}"]).Select(v => (int)v).ToArray()));

        using CpuBackend backend = new();
        Stopwatch watch = Stopwatch.StartNew();
        using XPartConditioner conditioner = new(backend, weights);
        output.WriteLine($"loaded in {watch.Elapsed.TotalSeconds:F1} s");
        watch.Restart();
        float[] sonata = conditioner.SonataFeatures(points, feats);
        output.WriteLine($"sonata features for {points.Length / 3} points: {watch.Elapsed.TotalSeconds:F1} s");
        Compare("sonata features", sonata, Values(fixture["sonata"]), 1e-3);

        watch.Restart();
        XPartConditioning result = conditioner.Condition(points, feats, objectQueries, parts);
        output.WriteLine($"conditioning (object + 2 parts, {objectQueries.Length} queries): {watch.Elapsed.TotalSeconds:F1} s");
        using Tensor objectTokens = result.ObjectTokens;
        Compare("obj_cond", Values(objectTokens), Values(fixture["obj_cond"]), 1e-3);
        for (int k = 0; k < parts.Count; k++)
        {
            using Tensor tokens = result.PartTokens[k];
            Compare($"geo_cond part {k}", Values(tokens), Values(fixture[$"geo_cond.{k}"]), 1e-3);
        }
    }
}
