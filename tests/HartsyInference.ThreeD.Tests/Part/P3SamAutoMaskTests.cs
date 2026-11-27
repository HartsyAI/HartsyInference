using HartsyInference.Core.Backends.Cpu;
using HartsyInference.ThreeD.Models.Hunyuan3D.Part;

namespace HartsyInference.ThreeD.Tests.Part;

/// <summary>Unit lane: the surface sampling and normalisation of the auto mask, and a whole run on tiny random weights.</summary>
public sealed class P3SamAutoMaskTests
{
    // Two flat triangles in the xy plane: face 0 has area 0.5 and face 1 (vertices (2,0), (2,3), (5,5)) has area 4.5, so face 1 should receive nine times the points.
    private static readonly float[] Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 2, 3, 0, 5, 5, 0];
    private static readonly int[] Faces = [0, 1, 2, 3, 4, 5];

    [Fact]
    public void Sample_WeightsFacesByAreaAndKeepsPointsOnTheirFace()
    {
        (double[] points, float[] normals, int[] face) = P3SamAutoMask.Sample(Vertices, Faces, 8000, new Random(3));

        double area0 = 0.5, area1 = 4.5;
        double share1 = face.Count(f => f == 1) / 8000.0;
        Assert.InRange(share1, area1 / (area0 + area1) - 0.03, area1 / (area0 + area1) + 0.03);
        Assert.All(Enumerable.Range(0, 8000), i =>
        {
            Assert.Equal(0.0, points[i * 3 + 2], 9);
            Assert.Equal(1f, Math.Abs(normals[i * 3 + 2]), 4);
        });
    }

    [Fact]
    public void Normalize_CentersOnTheBoxAndScalesTheLargestHalfExtentToOne()
    {
        double[] points = [2, 10, 4, 6, 12, 5, 4, 11, 4.5];

        double[] unit = P3SamAutoMask.Normalize(points, 3);

        Assert.Equal(-1.0, unit[0], 6);
        Assert.Equal(1.0, unit[3], 6);
        Assert.Equal(0.0, unit[1] + unit[4], 6); // y centred
        Assert.InRange(unit.Max(Math.Abs), 0.999, 1.0);
    }

    [Fact]
    public void Run_OnTinyRandomWeightsLabelsEveryFaceAndIsDeterministic()
    {
        HartsyInference.ThreeD.Models.Utonia.UtoniaConfig config = P3SamFeaturesTests.TinyNoRope();
        Dictionary<string, HartsyInference.Core.Tensors.Tensor> state = P3SamFeaturesTests.TinyState(config, 8 + 16 + 24);
        foreach ((string key, HartsyInference.Core.Tensors.Tensor value) in P3SamHeadsTests.Weights(4, featureDim: 8))
            state[key] = value;
        using CpuBackend backend = new();
        using P3SamFeatures features = new(backend, state, config);
        using P3SamHeads heads = new(state);
        P3SamAutoMask segmenter = new(backend, features, heads);
        List<string> log = [];
        segmenter.StageLog = log.Add;
        P3SamAutoMask.Options options = new() { PointCount = 300, PromptCount = 8, Seed = 5 };

        // Two unit squares in the xy plane, one at x 0 to 1 and one at x 3 to 4 (four triangles, separate vertices).
        float[] vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 3, 0, 0, 4, 0, 0, 3, 1, 0, 4, 1, 0];
        int[] faces = [0, 1, 2, 1, 3, 2, 4, 5, 6, 5, 7, 6];

        P3SamPartLabeler.Result a = segmenter.Run(vertices, faces, options);
        P3SamPartLabeler.Result b = segmenter.Run(vertices, faces, options);

        Assert.Equal(faces.Length / 3, a.FaceIds.Length);
        Assert.Equal(a.FaceIds, b.FaceIds);
        Assert.Equal(a.PartIds.Length, a.Boxes.Length);
        Assert.All(a.Boxes.SelectMany(box => box), v => Assert.True(float.IsFinite(v)));
        Assert.Contains(log, line => line.StartsWith("sonata features", StringComparison.Ordinal));
        Assert.Contains(log, line => line.StartsWith("masks for 8 prompts", StringComparison.Ordinal));
    }
}
