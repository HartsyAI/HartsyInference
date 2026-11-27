using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Models.Utonia;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Utonia;

public sealed class UtoniaSerializationTests
{
    [Fact]
    public void ZOrder_OnTwoByTwoByTwoGrid_SortsInMortonOrder()
    {
        List<(int X, int Y, int Z)> cells = Cube(2);

        List<(int X, int Y, int Z)> sorted = cells.OrderBy(c => UtoniaSerialization.ZOrderKey(c.X, c.Y, c.Z, 1)).ToList();

        Assert.Equal(new[] { (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1) }, sorted);
    }

    [Fact]
    public void Hilbert_OnTwoByTwoByTwoGrid_VisitsFaceAdjacentCellsInOrder()
    {
        List<(int X, int Y, int Z)> sorted = Cube(2)
            .OrderBy(c => UtoniaSerialization.HilbertKey(c.X, c.Y, c.Z, 1))
            .ToList();

        Assert.Equal(new[] { (0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1), (1, 0, 0) }, sorted);
    }

    [Fact]
    public void Hilbert_OnFourByFourByFourGrid_IsABijectionWhoseConsecutiveCellsAreAdjacent()
    {
        List<(int X, int Y, int Z)> cells = Cube(4);
        List<(long Code, (int X, int Y, int Z) Cell)> keyed = cells
            .Select(c => (UtoniaSerialization.HilbertKey(c.X, c.Y, c.Z, 2), c))
            .OrderBy(k => k.Item1)
            .ToList();

        Assert.Equal(Enumerable.Range(0, 64).Select(i => (long)i), keyed.Select(k => k.Code));
        for (int i = 1; i < keyed.Count; i++)
        {
            (int X, int Y, int Z) a = keyed[i - 1].Cell, b = keyed[i].Cell;
            int manhattan = Math.Abs(a.X - b.X) + Math.Abs(a.Y - b.Y) + Math.Abs(a.Z - b.Z);
            Assert.Equal(1, manhattan);
        }
    }

    [Fact]
    public void TransOrders_SwapXAndYBeforeEncoding()
    {
        foreach ((int x, int y, int z) in Cube(4))
        {
            Assert.Equal(
                UtoniaSerialization.ZOrderKey(y, x, z, 2),
                UtoniaSerialization.Key(UtoniaSerialOrder.ZOrderTrans, x, y, z, 2));
            Assert.Equal(
                UtoniaSerialization.HilbertKey(y, x, z, 2),
                UtoniaSerialization.Key(UtoniaSerialOrder.HilbertTrans, x, y, z, 2));
        }
    }

    [Fact]
    public void Serialize_KeepsBatchesContiguousAndInverseMatchesEachOrder()
    {
        int[] grid = { 3, 0, 1, 0, 2, 2, 1, 1, 0, 2, 0, 3, 0, 0, 0, 3, 3, 3 };
        int[] batch = { 0, 0, 0, 1, 1, 1 };

        UtoniaSerialized s = UtoniaSerialization.Serialize(grid, batch, new[] { UtoniaSerialOrder.ZOrder, UtoniaSerialOrder.Hilbert });

        for (int k = 0; k < 2; k++)
        {
            int[] order = s.Order[k], inverse = s.Inverse[k];
            Assert.Equal(Enumerable.Range(0, 6), order.OrderBy(i => i));
            for (int j = 0; j < order.Length; j++)
            {
                Assert.Equal(j, inverse[order[j]]);
            }

            Assert.All(order.Take(3), p => Assert.Equal(0, batch[p]));
            Assert.All(order.Skip(3), p => Assert.Equal(1, batch[p]));
        }
    }

    [Fact]
    public void Serialize_DepthIsBitLengthOfLargestCoordinatePlusOne()
    {
        UtoniaSerialized full = UtoniaSerialization.Serialize(new[] { 0, 0, 0, 1023, 5, 9 }, new[] { 0, 0 }, new[] { UtoniaSerialOrder.ZOrder });
        UtoniaSerialized small = UtoniaSerialization.Serialize(new[] { 0, 0, 0, 63, 5, 9 }, new[] { 0, 0 }, new[] { UtoniaSerialOrder.ZOrder });

        Assert.Equal(11, full.Depth);
        Assert.Equal(7, small.Depth);
    }

    [Fact]
    public void Serialize_RejectsDepthBeyondSixteenBits()
    {
        Assert.Throws<HartsyInferenceException>(() =>
            UtoniaSerialization.Serialize(new[] { 0, 0, 0, 1 << 16, 0, 0 }, new[] { 0, 0 }, new[] { UtoniaSerialOrder.ZOrder }));
    }

    [Fact]
    public void ParseOrder_MapsCheckpointNamesAndRejectsOthers()
    {
        Assert.Equal(UtoniaSerialOrder.ZOrder, UtoniaSerialization.ParseOrder("z"));
        Assert.Equal(UtoniaSerialOrder.ZOrderTrans, UtoniaSerialization.ParseOrder("z-trans"));
        Assert.Equal(UtoniaSerialOrder.Hilbert, UtoniaSerialization.ParseOrder("hilbert"));
        Assert.Equal(UtoniaSerialOrder.HilbertTrans, UtoniaSerialization.ParseOrder("hilbert-trans"));
        Assert.Throws<HartsyInferenceException>(() => UtoniaSerialization.ParseOrder("morton"));
    }

    private static List<(int X, int Y, int Z)> Cube(int size)
    {
        List<(int, int, int)> cells = new();
        for (int x = 0; x < size; x++)
        {
            for (int y = 0; y < size; y++)
            {
                for (int z = 0; z < size; z++)
                {
                    cells.Add((x, y, z));
                }
            }
        }

        return cells;
    }
}
