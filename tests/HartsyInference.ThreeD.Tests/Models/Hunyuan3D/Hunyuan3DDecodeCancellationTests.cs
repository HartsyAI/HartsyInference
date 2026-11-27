using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.Hunyuan3D;

namespace HartsyInference.ThreeD.Tests.Models.Hunyuan3D;

/// <summary>A cancelled shape decode stops within one query chunk (dense grid) or one query (FlashVDM) and leaves no result.</summary>
public sealed class Hunyuan3DDecodeCancellationTests
{
    [Fact]
    public void DecodeGrid_FillsTheGridXFastestInChunksAndStopsAtACancelledToken()
    {
        int calls = 0;
        ScalarField3D field = Hunyuan3DShapeVae.DecodeGrid(5, 1f, 40, (coords, count, logits) =>
        {
            calls++;
            for (int i = 0; i < count; i++)
                logits[i] = coords[i * 3] + 10 * coords[i * 3 + 1] + 100 * coords[i * 3 + 2];
        });

        Assert.Equal(4, calls); // 125 points in chunks of 40: 40, 40, 40, 5
        Assert.Equal(-1f + 10 * -1f + 100 * -1f, field.At(0, 0, 0));
        Assert.Equal(0.5f + 10 * 0f + 100 * -0.5f, field.At(3, 2, 1), 4);

        using CancellationTokenSource source = new();
        calls = 0;
        Assert.Throws<OperationCanceledException>(() => Hunyuan3DShapeVae.DecodeGrid(5, 1f, 40, (coords, count, logits) =>
        {
            if (++calls == 2)
                source.Cancel();
        }, source.Token));
        Assert.Equal(2, calls); // the chunk in flight finishes, the next one never starts
    }

    [Fact]
    public void DecodeGrid_WithACancelledTokenNeverQueries()
    {
        using CancellationTokenSource source = new();
        source.Cancel();
        int calls = 0;

        Assert.Throws<OperationCanceledException>(() => Hunyuan3DShapeVae.DecodeGrid(8, 1f, 64, (coords, count, logits) => calls++, source.Token));

        Assert.Equal(0, calls);
    }

    [Fact]
    public void FlashVdmDecode_StopsAtACancelledTokenBeforeTheNextQuery()
    {
        using CancellationTokenSource source = new();
        int queries = 0;

        Assert.Throws<OperationCanceledException>(() => Hunyuan3DFlashVdm.Decode((coordinates, count, stride, logits) =>
        {
            if (++queries == 3)
                source.Cancel();
            logits[..count].Clear();
        }, 128, 1.01f, 0f, cancel: source.Token));

        Assert.Equal(3, queries);
    }

    [Fact]
    public void FlashVdmDecode_WithACancelledTokenNeverQueries()
    {
        using CancellationTokenSource source = new();
        source.Cancel();
        int queries = 0;

        Assert.Throws<OperationCanceledException>(() => Hunyuan3DFlashVdm.Decode((coordinates, count, stride, logits) => queries++, 128, 1.01f, 0f, cancel: source.Token));

        Assert.Equal(0, queries);
    }
}
