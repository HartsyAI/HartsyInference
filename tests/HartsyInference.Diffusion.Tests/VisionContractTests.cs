using System.Threading.Channels;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The P2 vision contracts: the per-mode validation table, the float-array encodings the CLI and API share, and
/// the bounded frame stream's buffering, cancellation and cleanup.</summary>
public sealed class VisionContractTests
{
    // ── Validation: single-image requests ────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(VisionMode.Embed)]
    [InlineData(VisionMode.Depth)]
    [InlineData(VisionMode.MetricDepth)]
    [InlineData(VisionMode.Geometry)]
    [InlineData(VisionMode.Pose)]
    [InlineData(VisionMode.Matting)]
    [InlineData(VisionMode.Face)]
    [InlineData(VisionMode.Intrinsic)]
    public void SingleImage_ModesAcceptAPlainRequest(VisionMode mode)
    {
        VisionRequestValidator.Validate(new VisionRequest { Image = Image(8, 6), Mode = mode });
    }

    [Theory]
    [InlineData(VisionMode.Track)]
    [InlineData(VisionMode.Interpolate)]
    public void SingleImage_SequenceModesAreRefused(VisionMode mode)
    {
        ArgumentException ex = Assert.Throws<ArgumentException>(() =>
            VisionRequestValidator.Validate(new VisionRequest { Image = Image(8, 6), Mode = mode }));
        Assert.Contains("needs a frame sequence", ex.Message);
    }

    [Fact]
    public void SingleImage_TargetSizeOnlyForUpscale()
    {
        VisionRequestValidator.Validate(new VisionRequest { Image = Image(8, 6), Mode = VisionMode.Upscale, TargetWidth = 32 });
        ArgumentException ex = Assert.Throws<ArgumentException>(() =>
            VisionRequestValidator.Validate(new VisionRequest { Image = Image(8, 6), Mode = VisionMode.Depth, TargetWidth = 32 }));
        Assert.Contains("only applies to Upscale", ex.Message);
    }

    [Fact]
    public void SingleImage_RgbMustMatchTheSize()
    {
        ImageData bad = new ImageData { Rgb = new byte[10], Width = 8, Height = 6 };
        Assert.Throws<ArgumentException>(() => VisionRequestValidator.Validate(new VisionRequest { Image = bad, Mode = VisionMode.Embed }));
    }

    // ── Validation: batch requests ───────────────────────────────────────────────────────────────────

    public static TheoryData<string, VisionBatchRequest, string?> BatchCases => new()
    {
        { "views ok", Views(VisionMode.Geometry, 3), null },
        { "views on a frames-only mode", Views(VisionMode.Track, 3), "does not take several views" },
        { "views on a non-geometry mode", Views(VisionMode.Pose, 3), "does not take several views" },
        { "one view", Views(VisionMode.Depth, 1), "at least 2 views" },
        { "views with fps", Views(VisionMode.Depth, 2) with { Fps = 30 }, "no time order" },
        { "frames for a single-image mode", Frames(VisionMode.Depth, 3) with { Fps = 30 }, "does not take a frame sequence" },
        { "frames without timing", Frames(VisionMode.Interpolate, 3) with { InterpolationFactor = 2 }, "exactly one of TimestampsSeconds and Fps" },
        { "frames with both timings", Frames(VisionMode.Interpolate, 3) with { Fps = 30, TimestampsSeconds = [0, 1, 2], InterpolationFactor = 2 }, "exactly one of TimestampsSeconds and Fps" },
        { "timestamps out of order", Frames(VisionMode.Interpolate, 3) with { TimestampsSeconds = [0, 0.2, 0.1], InterpolationFactor = 2 }, "strictly increase" },
        { "timestamps miscounted", Frames(VisionMode.Interpolate, 3) with { TimestampsSeconds = [0, 0.1], InterpolationFactor = 2 }, "one per frame" },
        { "interpolate ok by factor", Frames(VisionMode.Interpolate, 3) with { Fps = 24, InterpolationFactor = 2 }, null },
        { "interpolate ok by target fps", Frames(VisionMode.Interpolate, 3) with { TimestampsSeconds = [0, 0.04, 0.08], TargetFps = 60 }, null },
        { "interpolate with both", Frames(VisionMode.Interpolate, 3) with { Fps = 24, InterpolationFactor = 2, TargetFps = 60 }, "exactly one of InterpolationFactor and TargetFps" },
        { "interpolate below source rate", Frames(VisionMode.Interpolate, 3) with { Fps = 30, TargetFps = 24 }, "above the source rate" },
        { "interpolate factor too large", Frames(VisionMode.Interpolate, 3) with { Fps = 30, InterpolationFactor = 64 }, "must be 2 to" },
        { "interpolate with prompts", Frames(VisionMode.Interpolate, 3) with { Fps = 30, InterpolationFactor = 2, Prompts = [Text(0)] }, "takes no prompts" },
        { "factor on track", Frames(VisionMode.Track, 3) with { Fps = 30, Prompts = [Text(0)], InterpolationFactor = 2 }, "only applies to Interpolate" },
        { "track ok", Frames(VisionMode.Track, 3) with { Fps = 30, Prompts = [Text(0)] }, null },
        { "track without prompts", Frames(VisionMode.Track, 3) with { Fps = 30 }, "at least one prompt" },
        { "track with an empty prompt", Frames(VisionMode.Track, 3) with { Fps = 30, Prompts = [new VisionFramePrompt { FrameIndex = 0 }] }, "at least one prompt" },
        { "track prompt past the end", Frames(VisionMode.Track, 3) with { Fps = 30, Prompts = [Text(3)] }, "names frame 3" },
        { "track point outside", Frames(VisionMode.Track, 3) with { Fps = 30, Prompts = [new VisionFramePrompt { FrameIndex = 0, Points = [new VisionPoint(9, 1)] }] }, "outside" },
        { "track empty box", Frames(VisionMode.Track, 3) with { Fps = 30, Prompts = [new VisionFramePrompt { FrameIndex = 0, Boxes = [new VisionBox(4, 1, 2, 3)] }] }, "empty or outside" },
        { "mixed sizes", Views(VisionMode.Depth, 2) with { Images = [Image(8, 6), Image(6, 8)] }, "share one size" },
    };

    [Theory]
    [MemberData(nameof(BatchCases))]
    public void Batch_ValidationTable(string name, VisionBatchRequest request, string? expectedError)
    {
        _ = name;
        if (expectedError is null)
        {
            VisionRequestValidator.Validate(request);
            return;
        }
        ArgumentException ex = Assert.Throws<ArgumentException>(() => VisionRequestValidator.Validate(request));
        Assert.Contains(expectedError, ex.Message);
    }

    [Fact]
    public void Batch_FrameTimesComeFromTimestampsOrFps()
    {
        Assert.Equal(0.5, (Frames(VisionMode.Interpolate, 3) with { Fps = 4 }).TimeOf(2));
        Assert.Equal(0.3, (Frames(VisionMode.Interpolate, 3) with { TimestampsSeconds = [0, 0.1, 0.3] }).TimeOf(2));
        Assert.Null(Views(VisionMode.Depth, 2).TimeOf(1));
    }

    // ── Array encodings ──────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Array_ShapeMustCoverTheData()
    {
        Assert.Throws<ArgumentException>(() => new VisionArray(new float[6], 2, 2));
        Assert.Throws<ArgumentException>(() => new VisionArray(new float[0], 0, 2));
        VisionArray ok = new VisionArray(new float[12], 2, 2, 3);
        Assert.Equal((2, 2), (ok.Height, ok.Width));
    }

    [Fact]
    public void Npy_RoundTripsAndCarriesTheSamePayloadAsTheWireBytes()
    {
        VisionArray array = new VisionArray([0f, -1.5f, 3.25f, float.MaxValue, 1e-30f, -0f], 2, 1, 3);

        byte[] npy = VisionArrayCodec.ToNpy(array);
        byte[] wire = VisionArrayCodec.ToLittleEndianBytes(array);

        int headerLength = BitConverter.ToUInt16(npy, 8);
        Assert.Equal(0, (10 + headerLength) % 64);
        Assert.Equal(wire, npy[(10 + headerLength)..]);
        VisionArray back = VisionArrayCodec.FromNpy(npy);
        Assert.Equal(array.Shape, back.Shape);
        Assert.Equal(array.Data, back.Data);
        Assert.Equal(array.Data, VisionArrayCodec.FromLittleEndianBytes(wire, array.Shape).Data);
    }

    [Fact]
    public void Npy_HeaderIsNumPysFormat()
    {
        byte[] npy = VisionArrayCodec.ToNpy(new VisionArray([1f, 2f, 3f], 3));
        string header = System.Text.Encoding.ASCII.GetString(npy, 10, BitConverter.ToUInt16(npy, 8));

        Assert.StartsWith("{'descr': '<f4', 'fortran_order': False, 'shape': (3,), }", header);
        Assert.EndsWith("\n", header);
    }

    // ── Frame stream ─────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public async Task Stream_DeliversEveryFrameInOrderAndDisposesTheState()
    {
        State state = new State();
        List<int> seen = [];
        await foreach (VisionFrameResult frame in VisionFrameStream.Run(Producer(state, 10), state))
            seen.Add(frame.Index);

        Assert.Equal(Enumerable.Range(0, 10), seen);
        Assert.True(state.Disposed);
        Assert.True(state.ProducerFinished);
    }

    [Fact]
    public async Task Stream_BreakingOutEarlyStopsTheProducerAndDisposes()
    {
        State state = new State();
        await foreach (VisionFrameResult frame in VisionFrameStream.Run(Producer(state, 1000), state, capacity: 2))
        {
            if (frame.Index == 1)
                break;
        }

        Assert.True(state.Disposed);
        Assert.True(state.ProducerFinished);
        Assert.True(state.Produced < 1000, $"the producer ran to {state.Produced} after the consumer stopped");
    }

    [Fact]
    public async Task Stream_ProducerStaysWithinTheBuffer()
    {
        State state = new State();
        await using IAsyncEnumerator<VisionFrameResult> e = VisionFrameStream.Run(Producer(state, 100), state, capacity: 3).GetAsyncEnumerator();
        Assert.True(await e.MoveNextAsync());
        await Task.Delay(200);

        // One frame taken, three buffered, one blocked in WriteAsync.
        Assert.InRange(state.Produced, 1, 5);
    }

    [Fact]
    public async Task Stream_CancellationStopsAndDisposes()
    {
        State state = new State();
        using CancellationTokenSource cts = new CancellationTokenSource();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (VisionFrameResult frame in VisionFrameStream.Run(Producer(state, 1000), state, cancel: cts.Token))
            {
                if (frame.Index == 2)
                    await cts.CancelAsync();
            }
        });

        Assert.True(state.Disposed);
        Assert.True(state.ProducerFinished);
    }

    [Fact]
    public async Task Stream_ProducerErrorReachesTheConsumerAndDisposes()
    {
        State state = new State();
        InvalidOperationException ex = await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (VisionFrameResult _ in VisionFrameStream.Run(Producer(state, 5, failAt: 3), state))
            {
            }
        });

        Assert.Equal("model failed at frame 3", ex.Message);
        Assert.True(state.Disposed);
    }

    private static Func<ChannelWriter<VisionFrameResult>, CancellationToken, Task> Producer(State state, int count, int failAt = -1) =>
        async (writer, cancel) =>
        {
            try
            {
                for (int i = 0; i < count; i++)
                {
                    if (i == failAt)
                        throw new InvalidOperationException($"model failed at frame {i}");
                    await writer.WriteAsync(new VisionFrameResult { Index = i, TimestampSeconds = i / 30.0, Result = new VisionResult() }, cancel);
                    Interlocked.Increment(ref state.Produced);
                }
            }
            finally
            {
                state.ProducerFinished = true;
            }
        };

    private sealed class State : IDisposable
    {
        public int Produced;
        public volatile bool ProducerFinished;
        public bool Disposed { get; private set; }

        public void Dispose()
        {
            Assert.True(ProducerFinished, "state was disposed while the producer was still running");
            Disposed = true;
        }
    }

    private static ImageData Image(int width, int height) => new ImageData { Rgb = new byte[width * height * 3], Width = width, Height = height };

    private static VisionBatchRequest Views(VisionMode mode, int count) => new VisionBatchRequest
    {
        Images = Enumerable.Range(0, count).Select(_ => Image(8, 6)).ToList(),
        Mode = mode,
        Kind = VisionSequenceKind.Views,
    };

    private static VisionBatchRequest Frames(VisionMode mode, int count) => Views(mode, count) with { Kind = VisionSequenceKind.Frames };

    private static VisionFramePrompt Text(int frame) => new VisionFramePrompt { FrameIndex = frame, ObjectId = 1, Text = "the dog" };
}
