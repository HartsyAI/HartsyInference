using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.HairDeck;

namespace HartsyInference.ThreeD.Tests;

public sealed class HairDeckGeometryTests
{
    private static readonly float[] HairSrgb = [0.4f, 0.3f, 0.25f];

    [Fact]
    public void Root_normal_sets_the_width_direction_of_a_planar_card()
    {
        DecodedCard card = Arc(4, rootZ: 0f, rootNormal: [0f, 0f, 1f]);

        HairCardMesh? mesh = HairDeckGeometry.BuildCards(new HairDeckResult([card], [], 0), HairSrgb);

        // corner 0 of ring 0 is root + (width / 2) * normal; the tangent is in the plane, so the normal is kept.
        float[] v = mesh!.Mesh.Vertices;
        Assert.Equal(0f, v[0] - card.Centerline[0], 6);
        Assert.Equal(0f, v[1] - card.Centerline[1], 6);
        Assert.Equal(0.01f, v[2] - card.Centerline[2], 6);
    }

    [Fact]
    public void Root_normal_sign_is_used_as_given()
    {
        DecodedCard card = Arc(4, rootZ: 0f, rootNormal: [0f, 0f, -1f]);

        HairCardMesh? mesh = HairDeckGeometry.BuildCards(new HairDeckResult([card], [], 0), HairSrgb);

        Assert.Equal(-0.01f, mesh!.Mesh.Vertices[2] - card.Centerline[2], 6);
    }

    [Fact]
    public void Root_normal_of_an_oblique_planar_card_is_kept_in_its_plane_normal_direction()
    {
        // Card in the plane spanned by u = (1, 1, 0) / sqrt 2 and v = (0, 1, 1) / sqrt 2; its normal is (1, -1, 1).
        double s2 = Math.Sqrt(2.0);
        double[] st = [0.0, 0.0, 0.01, 0.002, 0.02, 0.0, 0.03, 0.01];
        float[] centreline = new float[12];
        for (int i = 0; i < 4; i++)
        {
            double sv = st[2 * i];
            double tv = st[2 * i + 1];
            centreline[3 * i] = (float)(sv / s2);
            centreline[3 * i + 1] = (float)((sv + tv) / s2);
            centreline[3 * i + 2] = (float)(tv / s2);
        }
        float s3 = (float)Math.Sqrt(3.0);
        DecodedCard card = Card(centreline, 4, [1f / s3, -1f / s3, 1f / s3]);

        HairCardMesh? mesh = HairDeckGeometry.BuildCards(new HairDeckResult([card], [], 0), HairSrgb);

        float[] v = mesh!.Mesh.Vertices;
        double[] offset = [v[0] - centreline[0], v[1] - centreline[1], v[2] - centreline[2]];
        double[] normal = [1.0 / Math.Sqrt(3.0), -1.0 / Math.Sqrt(3.0), 1.0 / Math.Sqrt(3.0)];
        double dot = offset[0] * normal[0] + offset[1] * normal[1] + offset[2] * normal[2];
        double length = Math.Sqrt(offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]);
        Assert.Equal(0.01, length, 6);
        Assert.Equal(1.0, Math.Abs(dot) / length, 5);
    }

    [Fact]
    public void Per_point_width_profile_drives_each_ring()
    {
        float[] widths = [0.02f, 0.01f, 0.03f, 0.005f];
        DecodedCard card = Arc(4, rootZ: 0f, rootNormal: [0f, 0f, 1f]) with { WidthProfile = widths };

        HairCardMesh? mesh = HairDeckGeometry.BuildCards(new HairDeckResult([card], [], 0), HairSrgb);

        float[] v = mesh!.Mesh.Vertices;
        for (int ring = 0; ring < 4; ring++)
        {
            int corner0 = 4 * ring;
            Assert.Equal(widths[ring] / 2f, v[3 * corner0 + 2] - card.Centerline[3 * ring + 2], 6);
        }
    }

    [Fact]
    public void Root_normal_is_required_and_must_not_be_zero()
    {
        DecodedCard missing = Arc(4, rootZ: 0f, rootNormal: []);
        DecodedCard zero = Arc(4, rootZ: 0f, rootNormal: [0f, 0f, 0f]);

        Assert.Throws<ArgumentException>(() => HairDeckGeometry.BuildCards(new HairDeckResult([missing], [], 0), HairSrgb));
        Assert.Throws<ArgumentException>(() => HairDeckGeometry.BuildCards(new HairDeckResult([zero], [], 0), HairSrgb));
    }

    [Fact]
    public void Profiles_must_hold_one_value_per_point()
    {
        DecodedCard card = Arc(4, rootZ: 0f, rootNormal: [0f, 0f, 1f]) with { ThicknessProfile = [0.004f, 0.004f] };

        Assert.Throws<ArgumentException>(() => HairDeckGeometry.BuildCards(new HairDeckResult([card], [], 0), HairSrgb));
    }

    [Fact]
    public void Without_colours_the_base_colour_is_the_hair_colour()
    {
        DecodedCard card = Arc(4, rootZ: 0f, rootNormal: [0f, 0f, 1f]);

        HairCardMesh? mesh = HairDeckGeometry.BuildCards(new HairDeckResult([card], [], 0), HairSrgb);

        Assert.NotNull(mesh);
        Assert.Equal(0.132868, mesh.BaseColorLinear[0], 5);
        Assert.Equal(0.9388011154765376f, mesh.Mesh.VertexColors![0], 5);
    }

    [Fact]
    public void Card_colours_are_passed_through_and_the_base_colour_turns_white()
    {
        float[] bins = new float[16 * 3];
        Array.Fill(bins, 0.5f);
        DecodedCard card = Card(Arc(4, rootZ: 0f, rootNormal: [0f, 0f, 1f]).Centerline, 4, [0f, 0f, 1f], bins);

        HairCardMesh? mesh = HairDeckGeometry.BuildCards(new HairDeckResult([card], [], 0), HairSrgb);

        Assert.Equal(new[] { 1f, 1f, 1f, 1f }, mesh!.BaseColorLinear);
        Assert.Equal(0.9388011154765376f * 0.5f, mesh.Mesh.VertexColors![0], 5);
    }

    [Fact]
    public void Cards_with_fewer_than_three_points_are_skipped_and_no_meshable_card_gives_null()
    {
        DecodedCard two = Card([0f, 0f, 0f, 0.01f, 0f, 0f], 2, [0f, 0f, 1f]);

        Assert.Null(HairDeckGeometry.BuildCards(new HairDeckResult([two], [], 0), HairSrgb));
    }

    [Fact]
    public void Centreline_length_must_match_the_point_count()
    {
        DecodedCard bad = Card([0f, 0f, 0f, 1f, 0f, 0f], 3, [0f, 0f, 1f]);

        Assert.Throws<ArgumentException>(() => HairDeckGeometry.BuildCards(new HairDeckResult([bad], [], 0), HairSrgb));
    }

    [Fact]
    public void Chains_use_joint_count_and_ignore_the_path()
    {
        DecodedChain chain = new([0f, 0f, 0f, 0f, 0f, 0.1f, 9f, 9f, 9f], 2, 0.5f);

        Mesh? mesh = HairDeckGeometry.BuildChains(new HairDeckResult([], [chain], 0));

        Assert.NotNull(mesh);
        Assert.Equal(18 + 2 * 42, mesh.VertexCount);
    }

    [Fact]
    public void Joint_count_larger_than_the_joint_array_is_rejected()
    {
        DecodedChain chain = new([0f, 0f, 0f], 2, 0.5f);

        Assert.Throws<ArgumentException>(() => HairDeckGeometry.BuildChains(new HairDeckResult([], [chain], 0)));
    }

    /// <summary>Card with the given centreline; width and thickness profiles are constant.</summary>
    private static DecodedCard Card(float[] centreline, int points, float[] rootNormal, float[]? colors = null)
    {
        float[] widths = new float[points];
        float[] thicknesses = new float[points];
        Array.Fill(widths, 0.02f);
        Array.Fill(thicknesses, 0.004f);
        return new DecodedCard(centreline, points, 0.02f, 0.004f, 0, colors)
        {
            WidthProfile = widths,
            ThicknessProfile = thicknesses,
            RootNormal = rootNormal,
        };
    }

    /// <summary>Points on a circular arc in the xy plane at height rootZ (planar).</summary>
    private static DecodedCard Arc(int points, float rootZ, float[] rootNormal)
    {
        float[] centreline = new float[points * 3];
        for (int i = 0; i < points; i++)
        {
            double angle = 0.2 * i;
            centreline[3 * i] = (float)(0.1 * Math.Cos(angle));
            centreline[3 * i + 1] = (float)(0.1 * Math.Sin(angle));
            centreline[3 * i + 2] = rootZ;
        }
        return Card(centreline, points, rootNormal);
    }
}
