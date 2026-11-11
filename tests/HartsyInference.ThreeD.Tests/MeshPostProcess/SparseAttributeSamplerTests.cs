using System.Numerics;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Tests.MeshPostProcess;

/// <summary>Unit lane, synthetic volumes: trilinear sampling against hand-computed values and the voxel-center convention.</summary>
public sealed class SparseAttributeSamplerTests
{
    // A full 2x2x2 block where voxel (x,y,z) holds x + 2y + 4z, a linear field that trilinear interpolation reproduces exactly.
    private static SparseAttributeVolume LinearBlock(Vector3 origin = default, float voxelSize = 1f)
    {
        List<int> coords = [];
        List<float> attrs = [];
        for (int z = 0; z < 2; z++)
            for (int y = 0; y < 2; y++)
                for (int x = 0; x < 2; x++)
                {
                    coords.AddRange([x, y, z]);
                    attrs.Add(x + 2 * y + 4 * z);
                }
        return new SparseAttributeVolume([.. coords], [.. attrs], 1, (4, 4, 4), origin, voxelSize);
    }

    [Theory]
    [InlineData(0.5, 0.5, 0.5, 0.0)]
    [InlineData(1.5, 0.5, 0.5, 1.0)]
    [InlineData(1.0, 1.0, 1.0, 3.5)]
    [InlineData(1.25, 0.75, 1.1, 0.75 + 2 * 0.25 + 4 * 0.6)]
    public void SampleGrid_InterpolatesBetweenVoxelCenters(double gx, double gy, double gz, double expected)
    {
        Span<float> value = stackalloc float[1];

        bool hit = SparseAttributeSampler.SampleGrid(LinearBlock(), gx, gy, gz, value);

        Assert.True(hit);
        Assert.Equal(expected, value[0], 5);
    }

    [Fact]
    public void SampleGrid_RenormalizesOverTheOccupiedNeighborsOnly()
    {
        SparseAttributeVolume single = new([1, 1, 1], [5f], 1, (4, 4, 4), Vector3.Zero, 1f);
        Span<float> value = stackalloc float[1];

        bool hit = SparseAttributeSampler.SampleGrid(single, 1.9, 1.1, 1.4, value);

        Assert.True(hit);
        Assert.Equal(5f, value[0], 5);
    }

    [Fact]
    public void SampleGrid_FarFromEveryVoxelReturnsFalseAndZeros()
    {
        Span<float> value = stackalloc float[1];
        value[0] = 9;

        bool hit = SparseAttributeSampler.SampleGrid(LinearBlock(), 3.7, 3.7, 3.7, value);

        Assert.False(hit);
        Assert.Equal(0f, value[0]);
    }

    [Fact]
    public void SampleGrid_FillsEveryChannel()
    {
        SparseAttributeVolume two = new([0, 0, 0, 1, 0, 0], [1f, 10f, 3f, 30f], 2, (2, 1, 1), Vector3.Zero, 1f);
        Span<float> value = stackalloc float[2];

        SparseAttributeSampler.SampleGrid(two, 1.0, 0.5, 0.5, value);

        Assert.Equal(2f, value[0], 5);
        Assert.Equal(20f, value[1], 5);
    }

    [Fact]
    public void SampleWorld_AppliesOriginAndVoxelSize()
    {
        SparseAttributeVolume volume = LinearBlock(new Vector3(-1f, 2f, 0.5f), 0.25f);
        Span<float> viaWorld = stackalloc float[1];
        Span<float> viaGrid = stackalloc float[1];

        SparseAttributeSampler.SampleWorld(volume, -1.0 + 0.25 * 1.25, 2.0 + 0.25 * 0.75, 0.5 + 0.25 * 1.1, viaWorld);
        SparseAttributeSampler.SampleGrid(volume, 1.25, 0.75, 1.1, viaGrid);

        Assert.Equal(viaGrid[0], viaWorld[0], 5);
    }

    [Fact]
    public void Constructor_RejectsDuplicatesAndOutOfGridVoxels()
    {
        Assert.Throws<ArgumentException>(() => new SparseAttributeVolume([0, 0, 0, 0, 0, 0], [1f, 2f], 1, (2, 2, 2), Vector3.Zero, 1f));
        Assert.Throws<ArgumentException>(() => new SparseAttributeVolume([2, 0, 0], [1f], 1, (2, 2, 2), Vector3.Zero, 1f));
        Assert.Throws<ArgumentException>(() => new SparseAttributeVolume([0, 0, 0], [1f, 2f], 1, (2, 2, 2), Vector3.Zero, 1f));
    }

    [Fact]
    public void TryFindIndex_FindsEveryVoxelAndNothingElse()
    {
        SparseAttributeVolume volume = LinearBlock();

        for (int i = 0; i < volume.Count; i++)
        {
            Assert.True(volume.TryFindIndex(volume.Coords[i * 3], volume.Coords[i * 3 + 1], volume.Coords[i * 3 + 2], out int found));
            Assert.Equal(i, found);
        }
        Assert.False(volume.TryFindIndex(3, 3, 3, out _));
        Assert.False(volume.TryFindIndex(-1, 0, 0, out _));
        Assert.False(volume.TryFindIndex(4, 0, 0, out _));
    }

    [Fact]
    public void Layout_ValidatesChannelRanges()
    {
        Assert.Equal(6, PbrAttributeLayout.Default.RequiredChannels);
        PbrAttributeLayout.Default.Validate(6);
        Assert.Throws<ArgumentException>(() => PbrAttributeLayout.Default.Validate(5));
        Assert.Equal(5, new PbrAttributeLayout(0, 3, 4, -1).RequiredChannels);
    }
}
