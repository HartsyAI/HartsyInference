using HartsyInference.ThreeD.Models.Hunyuan3D.Part;

namespace HartsyInference.ThreeD.Tests.Part;

/// <summary>Unit lane, synthetic meshes: the geometry half of P3-SAM's auto mask turns per-prompt masks into labelled faces and part boxes.</summary>
public sealed class P3SamPartLabelerTests
{
    // Axis-aligned cube with its own eight vertices; returns vertices and 12 outward triangles.
    private static (float[] Vertices, int[] Faces) Cube(float x, float size)
    {
        List<float> verts = [];
        for (int i = 0; i < 8; i++)
        {
            verts.Add(x + (i & 1) * size);
            verts.Add((i >> 1 & 1) * size);
            verts.Add((i >> 2 & 1) * size);
        }

        int[] faces = [0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5];
        return ([.. verts], faces);
    }

    private static (float[] Vertices, int[] Faces) Merge(params (float[] Vertices, int[] Faces)[] meshes)
    {
        List<float> vertices = [];
        List<int> faces = [];
        foreach ((float[] v, int[] f) in meshes)
        {
            int offset = vertices.Count / 3;
            vertices.AddRange(v);
            faces.AddRange(f.Select(i => i + offset));
        }

        return ([.. vertices], [.. faces]);
    }

    // One point per face (its centroid) and the face it came from.
    private static (float[] Points, int[] PointFace) Centroids(float[] vertices, int[] faces)
    {
        int count = faces.Length / 3;
        float[] points = new float[count * 3];
        for (int f = 0; f < count; f++)
            for (int c = 0; c < 3; c++)
                points[f * 3 + c] = (vertices[faces[f * 3] * 3 + c] + vertices[faces[f * 3 + 1] * 3 + c] + vertices[faces[f * 3 + 2] * 3 + c]) / 3f;
        return (points, [.. Enumerable.Range(0, count)]);
    }

    [Fact]
    public void Label_SplitsSeparateObjectsIntoPartsAndGivesAnUnmaskedOneItsOwnPart()
    {
        (float[] vertices, int[] faces) = Merge(Cube(0f, 1f), Cube(2f, 1f), Cube(4f, 1f));
        (float[] points, int[] pointFace) = Centroids(vertices, faces);
        // Four identical masks per cube for the first two cubes (a cluster of more than two survives), none for the third.
        List<bool[]> masks = [];
        List<float> iou = [];
        for (int cube = 0; cube < 2; cube++)
            for (int repeat = 0; repeat < 4; repeat++)
            {
                masks.Add([.. Enumerable.Range(0, points.Length / 3).Select(p => p / 12 == cube)]);
                iou.Add(0.95f - 0.01f * repeat - 0.1f * cube);
            }

        P3SamPartLabeler.Result result = P3SamPartLabeler.Label(vertices, faces, points, pointFace, [.. masks], [.. iou], postProcess: false);

        Assert.Equal(3, result.PartIds.Length);
        Assert.Equal(3, result.Boxes.Length);
        for (int cube = 0; cube < 3; cube++)
        {
            int label = result.FaceIds[cube * 12];
            Assert.True(label >= 0);
            Assert.All(Enumerable.Range(cube * 12, 12), f => Assert.Equal(label, result.FaceIds[f]));
        }

        Assert.Equal(3, result.FaceIds.Distinct().Count());
        float[] expectedMinX = [0f, 2f, 4f];
        foreach (int id in result.PartIds)
        {
            int cube = Array.IndexOf(result.FaceIds, id) / 12;
            float[] box = result.Boxes[Array.IndexOf(result.PartIds, id)];
            Assert.Equal(expectedMinX[cube], box[0], 4);
            Assert.Equal(expectedMinX[cube] + 1f, box[3], 4);
            Assert.Equal(0f, box[1], 4);
            Assert.Equal(1f, box[4], 4);
        }
    }

    [Fact]
    public void Label_MergesMasksThatCoverTheSameObjectAndFillsFacesWithoutAPoint()
    {
        (float[] vertices, int[] faces) = Cube(0f, 1f);
        (float[] points, int[] pointFace) = Centroids(vertices, faces);
        // The sampler missed faces 4 and 5: they have no point and take the label of their neighbours.
        int[] keep = [.. Enumerable.Range(0, 12).Where(f => f is not (4 or 5))];
        points = [.. keep.SelectMany(f => points.Skip(f * 3).Take(3))];
        pointFace = keep;
        bool[] all = [.. Enumerable.Repeat(true, keep.Length)];
        bool[] nearlyAll = [.. Enumerable.Repeat(true, keep.Length - 1), false]; // IoU 0.9 or more with the first: joins its cluster
        bool[][] masks = [all, all, nearlyAll, all];

        P3SamPartLabeler.Result result = P3SamPartLabeler.Label(vertices, faces, points, pointFace, masks, [0.9f, 0.8f, 0.7f, 0.6f], postProcess: true);

        Assert.Single(result.PartIds);
        Assert.All(result.FaceIds, id => Assert.Equal(result.PartIds[0], id));
    }

    [Fact]
    public void FixLabel_FillsFromTheMajorityOfLabelledNeighboursAndConnectedRegionsFollowLabels()
    {
        // A strip of six faces in a row: labels 0 0 -1 -1 1 1. Labels update in place within a pass (as upstream's loop does): face 2 takes 0, then face 3 sees
        // the neighbours 0 (just filled) and 1, a tie that goes to the lowest label.
        int[][] adjacency = [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4]];
        int[] ids = [0, 0, -1, -1, 1, 1];

        P3SamPartLabeler.FixLabel(ids, adjacency);

        Assert.Equal([0, 0, 0, 0, 1, 1], ids);
        List<List<int>> regions = P3SamPartLabeler.ConnectedRegions(ids, adjacency, out int[] partOf);
        Assert.Equal(2, regions.Count);
        Assert.Equal([0, 0, 0, 0, 1, 1], partOf);
    }
}
