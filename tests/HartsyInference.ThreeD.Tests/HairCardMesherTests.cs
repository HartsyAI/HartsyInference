using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Tests;

public sealed class HairCardMesherTests
{
    private static readonly float[] HairSrgb = [0.4f, 0.3f, 0.25f];

    [Fact]
    public void Square_card_has_four_vertices_per_point_and_ten_faces_per_ring()
    {
        HairCardMesh? result = HairCardMesher.Build([StraightCard(5, 0, 0.02, 0.004)], HairSrgb);

        Assert.NotNull(result);
        Assert.Equal(20 * 3, result.Mesh.Vertices.Length);
        Assert.Equal(42 * 3, result.Mesh.Indices.Length);
        Assert.Equal(20 * 2, result.Mesh.Uvs!.Length);
        Assert.Equal(20 * 3, result.Mesh.VertexColors!.Length);
    }

    [Fact]
    public void Triangle_card_has_the_same_topology_as_a_square_card()
    {
        HairCardMesh? result = HairCardMesher.Build([StraightCard(6, 1, 0.02, 0.006)], HairSrgb);

        Assert.NotNull(result);
        Assert.Equal(24 * 3, result.Mesh.Vertices.Length);
        Assert.Equal(52 * 3, result.Mesh.Indices.Length);
    }

    [Fact]
    public void Square_ring_corners_follow_width_and_thickness_in_the_reference_frame()
    {
        double width = 0.02;
        double thickness = 0.004;
        HairCardMesh? result = HairCardMesher.Build([StraightCard(5, 0, width, thickness)], HairSrgb);

        // Root tangent points from tip to root (-x here), width direction +y, so the binormal is +z.
        float[] v = result!.Mesh.Vertices;
        AssertVector([0.02f, (float)(width / 2), 0f], Corner(v, 2, 0));
        AssertVector([0.02f, 0f, (float)(thickness / 2)], Corner(v, 2, 1));
        AssertVector([0.02f, -(float)(width / 2), 0f], Corner(v, 2, 2));
        AssertVector([0.02f, 0f, -(float)(thickness / 2)], Corner(v, 2, 3));
    }

    [Fact]
    public void Triangle_section_uses_the_reference_thickness_split()
    {
        double thickness = 0.006;
        HairCardMesh? result = HairCardMesher.Build([StraightCard(5, 1, 0.02, thickness)], HairSrgb);

        float[] v = result!.Mesh.Vertices;
        Assert.Equal((float)(thickness / 3), Corner(v, 2, 1)[2], 7);
        Assert.Equal(-(float)(thickness / 6), Corner(v, 2, 3)[2], 7);
    }

    [Fact]
    public void Zero_width_drops_degenerate_faces()
    {
        HairCardMesh? result = HairCardMesher.Build([StraightCard(5, 0, 0.0, 0.004)], HairSrgb);

        Assert.NotNull(result);
        Assert.Equal(20 * 3, result.Mesh.Vertices.Length);
        Assert.True(result.Mesh.Indices.Length < 42 * 3);
        Assert.Equal(0, result.Mesh.Indices.Length % 3);
    }

    [Fact]
    public void Card_uvs_run_across_the_width_and_along_arc_length()
    {
        HairCardMesh? result = HairCardMesher.Build([StraightCard(5, 0, 0.02, 0.004)], HairSrgb);

        float[] uv = result!.Mesh.Uvs!;
        float[] expectedU = [0f, 0.5f, 1f, 0.5f];
        for (int ring = 0; ring < 5; ring++)
        {
            for (int corner = 0; corner < 4; corner++)
            {
                int index = 4 * ring + corner;
                Assert.Equal(expectedU[corner], uv[2 * index], 6);
                Assert.Equal(ring / 4f, uv[2 * index + 1], 6);
            }
        }
    }

    [Fact]
    public void Without_card_colours_colour_is_the_brightness_and_base_colour_is_the_hair_colour()
    {
        HairCardMesh? result = HairCardMesher.Build([StraightCard(5, 0, 0.02, 0.004)], HairSrgb);

        float[] colors = result!.Mesh.VertexColors!;
        for (int i = 0; i < colors.Length; i++)
            Assert.Equal(0.9388011154765376f, colors[i], 6);
        Assert.Equal(SrgbToLinear(0.4), result.BaseColorLinear[0], 6);
        Assert.Equal(SrgbToLinear(0.3), result.BaseColorLinear[1], 6);
        Assert.Equal(SrgbToLinear(0.25), result.BaseColorLinear[2], 6);
        Assert.Equal(1f, result.BaseColorLinear[3]);
    }

    [Fact]
    public void Brightness_draws_advance_once_per_meshed_card_only()
    {
        HairCardMesh? result = HairCardMesher.Build(
            [StraightCard(5, 0, 0.02, 0.004), StraightCard(2, 0, 0.02, 0.004), StraightCard(5, 0, 0.02, 0.004)],
            HairSrgb);

        float[] colors = result!.Mesh.VertexColors!;
        int firstCardVertices = 20;
        Assert.Equal(0.9388011154765376f, colors[0], 6);
        Assert.Equal(0.9207526595696089f, colors[3 * firstCardVertices], 6);
    }

    [Fact]
    public void Card_colours_interpolate_bins_in_srgb_and_replace_the_base_colour()
    {
        float[] bins = new float[16 * 3];
        for (int b = 0; b < 16; b++)
        {
            bins[3 * b] = b / 15f;
            bins[3 * b + 1] = 0.2f;
            bins[3 * b + 2] = 0.5f;
        }
        HairCard card = StraightCard(5, 0, 0.02, 0.004) with { Colors = bins };

        HairCardMesh? result = HairCardMesher.Build([card], HairSrgb);

        float brightness = 0.9388011154765376f;
        float[] colors = result!.Mesh.VertexColors!;
        Assert.Equal(brightness * bins[0], colors[0], 5);
        Assert.Equal(brightness * 0.2f, colors[1], 5);
        int lastRing = 4 * 4 * 3;
        Assert.Equal(brightness * bins[45], colors[lastRing], 5);
        Assert.Equal(new[] { 1f, 1f, 1f, 1f }, result.BaseColorLinear);
    }

    [Fact]
    public void Mixed_card_colours_are_rejected()
    {
        HairCard coloured = StraightCard(5, 0, 0.02, 0.004) with { Colors = new float[48] };

        Assert.Throws<ArgumentException>(() => HairCardMesher.Build([coloured, StraightCard(5, 0, 0.02, 0.004)], HairSrgb));
    }

    [Fact]
    public void Build_returns_null_when_no_card_has_three_points()
    {
        Assert.Null(HairCardMesher.Build([StraightCard(2, 0, 0.02, 0.004)], HairSrgb));
        Assert.Null(HairCardMesher.Build([], HairSrgb));
    }

    [Fact]
    public void Unknown_shape_class_is_rejected()
    {
        Assert.Throws<ArgumentException>(() => HairCardMesher.Build([StraightCard(5, 2, 0.02, 0.004)], HairSrgb));
    }

    /// <summary>Straight card along +x, 1 cm per point, root width direction +y.</summary>
    private static HairCard StraightCard(int points, int shape, double width, double thickness)
    {
        double[] seq = new double[points * 8];
        for (int i = 0; i < points; i++)
        {
            seq[8 * i] = 0.01 * i;
            seq[8 * i + 1] = 0.0;
            seq[8 * i + 2] = 0.0;
            seq[8 * i + 3] = width;
            seq[8 * i + 4] = thickness;
            seq[8 * i + 5] = 0.0;
            seq[8 * i + 6] = 1.0;
            seq[8 * i + 7] = 0.0;
        }
        return new HairCard(seq, shape, null);
    }

    private static float[] Corner(float[] vertices, int ring, int corner)
    {
        int index = 4 * ring + corner;
        return [vertices[3 * index], vertices[3 * index + 1], vertices[3 * index + 2]];
    }

    private static void AssertVector(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], 6);
    }

    private static double SrgbToLinear(double c) => c <= 0.04045 ? c / 12.92 : Math.Pow((c + 0.055) / 1.055, 2.4);
}
