using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.Hunyuan3D;

namespace HartsyInference.ThreeD.Tests.Models.Hunyuan3D;

public sealed class Hunyuan3DFlashVdmTests
{
    private const float Radius = 0.6f;
    private const float Bound = 1.01f;
    private const float Steepness = 10f;   // keeps the |logit| < 0.95 band thin, as it is for a real decoder

    [Theory]
    [InlineData(128, new[] { 63, 126 })]
    [InlineData(256, new[] { 63, 126, 252 })]
    [InlineData(32, new[] { 31 })]
    public void Resolutions_FollowTheUpstreamLevels(int octree, int[] expected)
    {
        Assert.Equal(expected, Hunyuan3DFlashVdm.Resolutions(octree));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void Resolutions_RefuseAResolutionTooSmallForOneBlock(int octree)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Hunyuan3DFlashVdm.Resolutions(octree));
    }

    [Fact]
    public void Resolutions_AcceptTheSmallestBlockedGrid()
    {
        Assert.Equal([3], Hunyuan3DFlashVdm.Resolutions(3));
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(2.5f)]
    public void Decode_EvaluatesEveryCellThatCrossesTheLevelItWasGivenAsItsBand(float iso)
    {
        // The pipeline passes -iso, so the refined band is centred on logit +iso, the level the surface is extracted at.
        ScalarField3D field = Hunyuan3DFlashVdm.Decode(SphereLogits, 128, Bound, -iso);

        int n = field.ResX, crossing = 0;
        for (int z = 0; z < n - 1; z++)
        {
            for (int y = 0; y < n - 1; y++)
            {
                for (int x = 0; x < n - 1; x++)
                {
                    float low = float.MaxValue, high = float.MinValue;
                    bool anyNaN = false;
                    for (int c = 0; c < 8; c++)
                    {
                        float v = field.At(x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1));
                        anyNaN |= float.IsNaN(v);
                        low = Math.Min(low, v);
                        high = Math.Max(high, v);
                    }

                    if (!anyNaN && low < iso && high >= iso) crossing++;
                    Assert.False(anyNaN && Straddles(field, x, y, z, iso), $"cell ({x},{y},{z}) crosses {iso} but holds unevaluated points");
                }
            }
        }

        Assert.True(crossing > 1000, $"only {crossing} cells cross the level");
    }

    private static bool Straddles(ScalarField3D field, int x, int y, int z, float iso)
    {
        (float px, float py, float pz) = field.WorldOf(x, y, z);
        float centre = Steepness * (Radius - MathF.Sqrt(px * px + py * py + pz * pz));
        return MathF.Abs(centre - iso) < 0.5f * Steepness * (field.Max.X - field.Min.X) / (field.ResX - 1);
    }

    private static void SphereLogits(ReadOnlySpan<float> coordinates, int count, int stride, Span<float> logits)
    {
        for (int i = 0; i < count; i++)
        {
            float x = coordinates[i * 3], y = coordinates[i * 3 + 1], z = coordinates[i * 3 + 2];
            logits[i] = Steepness * (Radius - MathF.Sqrt(x * x + y * y + z * z));
        }
    }
}
