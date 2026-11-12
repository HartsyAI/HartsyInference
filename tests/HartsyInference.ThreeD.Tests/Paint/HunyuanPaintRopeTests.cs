using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>Unit lane, hand-computed values: voxel pooling of position maps and the rotary angle tables of Paint 2.1's multiview attention.</summary>
public sealed class HunyuanPaintRopeTests
{
    [Fact]
    public void VoxelIndices_AverageForegroundPixelsPerCellAndSkipBackgroundAndNearWhitePixels()
    {
        // 4x4 pixels, 2x2 grid: cell (0,0) is uniform (51, 102, 153); cell (0,1) is background; cell (1,0) has two foreground pixels and two with a 255 channel;
        // cell (1,1) is uniform (255, 0, 0), which upstream treats as background because one channel equals 1.
        byte[] map = new byte[4 * 4 * 3];
        void Set(int x, int y, byte r, byte g, byte b) { int i = (y * 4 + x) * 3; map[i] = r; map[i + 1] = g; map[i + 2] = b; }
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                Set(x, y, 255, 255, 255);
        foreach ((int x, int y) in new[] { (0, 0), (1, 0), (0, 1), (1, 1) })
            Set(x, y, 51, 102, 153);
        Set(0, 2, 102, 0, 51); Set(1, 2, 204, 0, 51); Set(0, 3, 255, 10, 10); Set(1, 3, 10, 255, 10);
        foreach ((int x, int y) in new[] { (2, 2), (3, 2), (2, 3), (3, 3) })
            Set(x, y, 255, 0, 0);

        int[] indices = HunyuanPaintRope.VoxelIndices(map, 1, 4, 2, 11);

        Assert.Equal([2, 4, 6], indices[0..3]);
        Assert.Equal([0, 0, 0], indices[3..6]);
        Assert.Equal([6, 0, 2], indices[6..9]);   // means 0.6, 0 and 0.2 of the two foreground pixels, times 10
        Assert.Equal([0, 0, 0], indices[9..12]);
    }

    [Fact]
    public void VoxelIndices_ZeroACellWithTooFewForegroundPixels()
    {
        // 8x8 pixels in a 1x1 grid: threshold is 8 * 8 / 16 = 4; three foreground pixels fall below it.
        byte[] map = new byte[8 * 8 * 3];
        Array.Fill(map, (byte)255);
        for (int x = 0; x < 3; x++)
            map[x * 3] = map[x * 3 + 1] = map[x * 3 + 2] = 100;

        Assert.Equal([0, 0, 0], HunyuanPaintRope.VoxelIndices(map, 1, 8, 1, 64));
        for (int x = 3; x < 4; x++)
            map[x * 3] = map[x * 3 + 1] = map[x * 3 + 2] = 100;
        Assert.NotEqual([0, 0, 0], HunyuanPaintRope.VoxelIndices(map, 1, 8, 1, 64));
    }

    [Fact]
    public void Tables_GiveEachAxisItsOwnFrequenciesAndRepeatEveryAnglePerPair()
    {
        // Head dimension 16: x and y take 6 channels each (3 pairs), z takes 4 (2 pairs).
        (float[] cos, float[] sin) = HunyuanPaintRope.Tables([3, 0, 2], 16);

        double[] xFreq = [1.0, Math.Pow(10000, -2.0 / 6), Math.Pow(10000, -4.0 / 6)], zFreq = [1.0, Math.Pow(10000, -2.0 / 4)];
        for (int k = 0; k < 3; k++)
            foreach (int column in new[] { 2 * k, 2 * k + 1 })
            {
                Assert.Equal(Math.Cos(3 * xFreq[k]), cos[column], 5);
                Assert.Equal(Math.Sin(3 * xFreq[k]), sin[column], 5);
            }
        for (int column = 6; column < 12; column++)
        {
            Assert.Equal(1f, cos[column]);
            Assert.Equal(0f, sin[column]);
        }
        for (int k = 0; k < 2; k++)
            foreach (int column in new[] { 12 + 2 * k, 13 + 2 * k })
            {
                Assert.Equal(Math.Cos(2 * zFreq[k]), cos[column], 5);
                Assert.Equal(Math.Sin(2 * zFreq[k]), sin[column], 5);
            }
        Assert.Throws<ArgumentException>(() => HunyuanPaintRope.Tables([1, 2, 3], 12));
    }

    [Fact]
    public void VoxelIndices_PoolInHalfPrecisionAsUpstreamDoes()
    {
        // One view, 8 x 8 pixels, every channel 234: the mean is 234 / 255 = 0.9176; scaled by 127 in F32 it rounds to 117, but upstream rounds the
        // value, the sum, the mean and the product to F16 first and lands on 116 (checked against torch in tests/python-reference/dump_paint21_unet.py).
        byte[] positions = new byte[8 * 8 * 3];
        Array.Fill(positions, (byte)234);

        int[] indices = HunyuanPaintRope.VoxelIndices(positions, 1, 8, 1, 128);

        Assert.Equal([116, 116, 116], indices);
    }
}
