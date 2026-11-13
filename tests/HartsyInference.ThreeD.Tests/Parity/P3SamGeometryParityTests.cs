using HartsyInference.ThreeD.Models.Hunyuan3D.Part;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>P3-SAM's automatic-segmentation geometry (<see cref="P3SamPartLabeler"/>) against upstream's own <c>mesh_sam</c> of <c>auto_mask.py</c> on the CPU (<c>tests/python-reference/dump_p3sam_geometry.py</c>), run
/// with a fake network that returns deterministic ball masks: the same mesh, sampled points and per-prompt masks go in, and the face labels and part boxes of the two implementations are compared.</summary>
/// <remarks>Integration lane: needs <c>HARTSY_PARITY_DIR</c> holding <c>p3sam_geometry_fp32.npz</c> (a mug, ball radii 0.2, 0.45 and 0.9) and <c>p3sam_geometry_bust_fp32.npz</c> (the bust, radii 0.08, 0.2 and 0.4: <c>--radii 0.08,0.2,0.4</c>). No weights.</remarks>
[Trait("Category", "Integration")]
public sealed class P3SamGeometryParityTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData("p3sam_geometry_fp32.npz", false)]
    [InlineData("p3sam_geometry_fp32.npz", true)]
    [InlineData("p3sam_geometry_bust_fp32.npz", false)]
    [InlineData("p3sam_geometry_bust_fp32.npz", true)]
    public void FaceLabelsAndPartBoxesMatchUpstreamMeshSam(string fixtureName, bool postProcess)
    {
        string parityDir = Environment.GetEnvironmentVariable(ParityFixtures.ParityDirVariable) ?? string.Empty;
        string path = Path.Combine(parityDir, fixtureName);
        if (!File.Exists(path))
        {
            string reason = $"{ParityFixtures.ParityDirVariable} does not hold {fixtureName}";
            if (Environment.GetEnvironmentVariable(ParityFixtures.RequireWeightsVariable) == "1")
                throw new InvalidOperationException($"{ParityFixtures.RequireWeightsVariable}=1 but {reason}.");
            output.WriteLine($"SKIPPED: {reason}");
            return;
        }

        NpzArchive fixture = NpzArchive.Open(path);
        float[] vertices = fixture.Get("vertices").ToFloat32(), points = fixture.Get("points").ToFloat32(), iou = fixture.Get("iou").ToFloat32();
        int[] faces = fixture.Get("faces").ToInt32(), pointFace = fixture.Get("face_idx").ToInt32();
        NpzArray maskArray = fixture.Get("masks");
        int prompts = (int)maskArray.Shape[0], pointCount = (int)maskArray.Shape[1];
        bool[][] masks = new bool[prompts][]; // stored as bytes, one per point
        for (int m = 0; m < prompts; m++)
        {
            masks[m] = new bool[pointCount];
            for (int p = 0; p < pointCount; p++)
                masks[m][p] = maskArray.Data[(long)m * pointCount + p] != 0;
        }

        string suffix = postProcess ? "1" : "0";
        int[] expectedIds = fixture.Get($"face_ids_{suffix}").ToInt32();
        float[] expectedBoxes = fixture.Get($"aabb_{suffix}").ToFloat32();

        P3SamPartLabeler.Result result = P3SamPartLabeler.Label(vertices, faces, points, pointFace, masks, iou, postProcess, 0.95);

        int different = 0;
        for (int f = 0; f < expectedIds.Length; f++)
            if (result.FaceIds[f] != expectedIds[f])
                different++;
        int expectedParts = expectedBoxes.Length / 6;
        output.WriteLine($"{fixtureName} post_process={postProcess}: {faces.Length / 3} faces, {pointCount} points, {prompts} prompts; upstream {expectedParts} parts, here {result.PartIds.Length}; {different} faces carry another label");
        Assert.Equal(expectedIds.Length, result.FaceIds.Length);
        Assert.Equal(0, different);
        Assert.Equal(expectedParts, result.Boxes.Length);
        double worst = 0;
        for (int part = 0; part < expectedParts; part++)
            for (int c = 0; c < 6; c++)
                worst = Math.Max(worst, Math.Abs(result.Boxes[part][c] - expectedBoxes[part * 6 + c]));
        output.WriteLine($"largest box difference {worst:E2}");
        Assert.True(worst < 1e-5, $"largest box difference {worst}");
    }
}
