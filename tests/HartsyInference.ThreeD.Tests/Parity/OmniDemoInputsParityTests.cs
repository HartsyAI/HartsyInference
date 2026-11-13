using HartsyInference.ThreeD.Models.Hunyuan3D.Omni;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>How the Omni demo control files become encoder rows, against upstream's own <c>inference.py</c> steps (<c>tests/python-reference/dump_omni_demo_inputs.py</c>: trimesh, the script's
/// <c>normalize_mesh</c>, the -90 degree turn, the 81,920 surface samples, <c>generate_voxel</c>) on the files of <c>demos/</c>: point, pose and bbox exactly, voxel as the overlap of the occupied cells
/// (the two sides draw different surface samples).</summary>
/// <remarks>Integration lane: needs <c>HARTSY_PARITY_DIR</c> holding <c>omni_demo_inputs.npz</c>. No weights.</remarks>
[Trait("Category", "Integration")]
public sealed class OmniDemoInputsParityTests(ITestOutputHelper output)
{
    private const string Fixture = "omni_demo_inputs.npz";

    private NpzArchive? Open()
    {
        string parityDir = Environment.GetEnvironmentVariable(ParityFixtures.ParityDirVariable) ?? string.Empty;
        string path = Path.Combine(parityDir, Fixture);
        if (File.Exists(path))
            return NpzArchive.Open(path);
        string reason = $"{ParityFixtures.ParityDirVariable} does not hold {Fixture}";
        if (Environment.GetEnvironmentVariable(ParityFixtures.RequireWeightsVariable) == "1")
            throw new InvalidOperationException($"{ParityFixtures.RequireWeightsVariable}=1 but {reason}.");
        output.WriteLine($"SKIPPED: {reason}");
        return null;
    }

    [Fact]
    public void Point_PoseAndBbox_RowsEqualUpstreams()
    {
        if (Open() is not { } fixture)
            return;

        float[] raw = fixture.Get("point.raw").ToFloat32(), normalised = fixture.Get("point.normalised").ToFloat32();
        float[] point = OmniControlInputs.Point(raw);
        Assert.Equal(normalised.Length * 2, point.Length); // three values repeated to six, one row per point (the demo cloud has no duplicate points)
        double worst = 0;
        for (int i = 0; i < normalised.Length / 3; i++)
            for (int k = 0; k < 6; k++)
                worst = Math.Max(worst, Math.Abs(point[i * 6 + k] - normalised[i * 3 + k % 3]));
        output.WriteLine($"point: {raw.Length / 3} points, rows differ from upstream's normalised cloud by at most {worst:E2}");
        Assert.True(worst < 1e-6, $"point rows differ by {worst}");

        float[] bones = fixture.Get("pose.raw").ToFloat32();
        Assert.Equal(bones, OmniControlInputs.Pose(bones));
        float[] box = fixture.Get("bbox.values").ToFloat32();
        Assert.Equal([box[0], box[1], box[2], box[0], box[1], box[2]], OmniControlInputs.BoundingBox(box[0], box[1], box[2]));
        output.WriteLine($"pose: {bones.Length / 6} bones identical; bbox {box[0]}, {box[1]}, {box[2]} identical");
    }

    [Fact]
    public void Voxel_OccupiedCellsOverlapUpstreamsAlmostCompletely()
    {
        if (Open() is not { } fixture)
            return;

        float[] vertices = fixture.Get("voxel.vertices").ToFloat32(), expected = fixture.Get("voxel.centres").ToFloat32();
        int[] faces = fixture.Get("voxel.faces").ToInt32();
        float[] rows = OmniControlInputs.Voxel(vertices, faces, seed: 0);
        HashSet<(float, float, float)> ours = [.. Enumerable.Range(0, rows.Length / 6).Select(i => (rows[i * 6], rows[i * 6 + 1], rows[i * 6 + 2]))];
        HashSet<(float, float, float)> theirs = [.. Enumerable.Range(0, expected.Length / 3).Select(i => (expected[i * 3], expected[i * 3 + 1], expected[i * 3 + 2]))];
        int shared = ours.Intersect(theirs).Count();
        double iou = (double)shared / (ours.Count + theirs.Count - shared);
        output.WriteLine($"voxel: {ours.Count} occupied cells here, {theirs.Count} upstream, {shared} shared, intersection over union {iou:F4}");
        Assert.True(iou > 0.97, $"the occupied cells overlap by only {iou:P1}");
    }
}
