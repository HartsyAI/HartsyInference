using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The MetricDepth and multi-view routing and the focal-length rule, without weights: what reaches Depth Pro 2, what is refused by
/// name, and the error a missing checkpoint gives.</summary>
public sealed class DepthPro2VisionServiceTests
{
    private static ImageData Image() => new() { Rgb = new byte[8 * 6 * 3], Width = 8, Height = 6 };

    private static ModelSpec Spec(string id) =>
        new() { Requested = id, Modality = Modality.Vision, Catalog = ModelCatalog.Find(id), LocalPath = null };

    [Fact]
    public void AFocalLengthIsForMetricDepthOnlyAndMustBePositive()
    {
        VisionRequestValidator.Validate(new VisionRequest { Image = Image(), Mode = VisionMode.MetricDepth, FocalLengthPx = 500 });
        VisionRequest depth = new() { Image = Image(), Mode = VisionMode.Depth, FocalLengthPx = 500 };
        Assert.Contains("only applies to MetricDepth", Assert.Throws<ArgumentException>(() => VisionRequestValidator.Validate(depth)).Message);
        foreach (double bad in new[] { 0, -1, double.NaN, double.PositiveInfinity })
        {
            Assert.Throws<ArgumentException>(() =>
                VisionRequestValidator.Validate(new VisionRequest { Image = Image(), Mode = VisionMode.MetricDepth, FocalLengthPx = bad }));
        }

        VisionBatchRequest views = new() { Images = [Image(), Image()], Mode = VisionMode.MetricDepth, Kind = VisionSequenceKind.Views, FocalLengthPx = 500 };
        VisionRequestValidator.Validate(views);
        Assert.Throws<ArgumentException>(() => VisionRequestValidator.Validate(views with { FocalLengthPx = -3 }));
        Assert.Throws<ArgumentException>(() => VisionRequestValidator.Validate(views with { Mode = VisionMode.Depth }));
    }

    [Fact]
    public async Task MetricDepthOnAModelThatIsNotDepthPro2IsRefusedByName()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        NotSupportedException ex = await Assert.ThrowsAsync<NotSupportedException>(() =>
            engine.Vision.RunAsync(Spec("depth-anything"), new VisionRequest { Image = Image(), Mode = VisionMode.MetricDepth }));
        Assert.Contains("MetricDepth", ex.Message);
    }

    [Fact]
    public async Task ViewsReachDepthPro2ForMetricDepthOnly()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        VisionBatchRequest views = new() { Images = [Image(), Image()], Mode = VisionMode.MetricDepth, Kind = VisionSequenceKind.Views };

        // Depth Pro 2 accepts the views and then needs its checkpoint.
        FileNotFoundException missing = await Assert.ThrowsAsync<FileNotFoundException>(() => engine.Vision.RunViewsAsync(Spec("depthpro2-s"), views));
        Assert.Contains("depthpro2-s", missing.Message);
        await Assert.ThrowsAsync<NotSupportedException>(() => engine.Vision.RunViewsAsync(Spec("depth-anything"), views));
        await Assert.ThrowsAsync<NotSupportedException>(() => engine.Vision.RunViewsAsync(Spec("depthpro2-s"), views with { Mode = VisionMode.Depth }));
    }

    [Fact]
    public async Task AnImageOverTheTokenCapIsRefusedBeforeAnyLoad()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        ImageData uhd = new() { Rgb = new byte[3840 * 2160 * 3], Width = 3840, Height = 2160 };
        ArgumentException ex = await Assert.ThrowsAsync<ArgumentException>(() =>
            engine.Vision.RunAsync(Spec("depthpro2-s"), new VisionRequest { Image = uhd, Mode = VisionMode.MetricDepth }));
        Assert.Contains("Downscale", ex.Message);
    }

    [Fact]
    public void MetricDepthTakesFramesWithTheVideoOptionsAndNothingElseDoes()
    {
        VisionBatchRequest frames = new()
        {
            Images = [Image(), Image(), Image()], Mode = VisionMode.MetricDepth, Kind = VisionSequenceKind.Frames, Fps = 24,
            FrameWindow = 3, FrameOverlap = 1, FrameBuffering = VisionFrameBuffering.Buffered, FocalLengthPx = 300,
        };
        VisionRequestValidator.Validate(frames);
        VisionRequestValidator.Validate(frames with { FrameOverlap = null, FrameBuffering = VisionFrameBuffering.Unbuffered });
        VisionRequestValidator.Validate(frames with { FrameWindow = null, FrameOverlap = null, FrameBuffering = null });

        string Refused(VisionBatchRequest request) => Assert.Throws<ArgumentException>(() => VisionRequestValidator.Validate(request)).Message;
        Assert.Contains("exactly one of TimestampsSeconds and Fps", Refused(frames with { Fps = null }));
        Assert.Contains("FrameOverlap", Refused(frames with { FrameOverlap = 3 }));
        Assert.Contains("FrameOverlap", Refused(frames with { FrameOverlap = -1 }));
        Assert.Contains("do not overlap", Refused(frames with { FrameBuffering = VisionFrameBuffering.Unbuffered }));
        Assert.Contains("FrameWindow", Refused(frames with { FrameWindow = 0 }));
        Assert.Contains("FrameWindow", Refused(frames with { FrameWindow = VisionRequestValidator.MaxFrameWindow + 1, FrameOverlap = null }));
        Assert.Contains("Prompts only apply to Track", Refused(frames with { Prompts = [new VisionFramePrompt { FrameIndex = 0, Text = "a cup" }] }));
        Assert.Contains("only applies to MetricDepth frames", Refused(frames with { Kind = VisionSequenceKind.Views, Fps = null }));
        Assert.Contains("Frames are for Track, Interpolate and MetricDepth", Refused(frames with { Mode = VisionMode.Depth, FrameWindow = null,
            FrameOverlap = null, FrameBuffering = null, FocalLengthPx = null }));
        Assert.True(VisionRequestValidator.AcceptsFrames(VisionMode.MetricDepth));
        Assert.False(VisionRequestValidator.IsSequenceOnly(VisionMode.MetricDepth));
    }

    [Fact]
    public void FramesReachDepthPro2AndRefuseBeforeTheStreamStarts()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        VisionBatchRequest frames = new() { Images = [Image(), Image()], Mode = VisionMode.MetricDepth, Kind = VisionSequenceKind.Frames, Fps = 10 };

        // Each refusal is thrown by the call itself, so an API can turn it into a status before any frame is sent.
        FileNotFoundException missing = Assert.Throws<FileNotFoundException>(() => engine.Vision.RunFramesAsync(Spec("depthpro2-s"), frames));
        Assert.Contains("depthpro2-s", missing.Message);
        Assert.Throws<NotSupportedException>(() => engine.Vision.RunFramesAsync(Spec("depth-anything"), frames));
        // The default window is 16, so an overlap of 16 passes the validator (which knows no model default) and is refused by the model.
        Assert.Contains("below the frame window", Assert.Throws<ArgumentException>(() =>
            engine.Vision.RunFramesAsync(Spec("depthpro2-s"), frames with { FrameOverlap = 16 })).Message);
    }

    [Fact]
    public async Task ASingleImageWithoutTheCheckpointNamesTheRow()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        FileNotFoundException ex = await Assert.ThrowsAsync<FileNotFoundException>(() =>
            engine.Vision.RunAsync(Spec("depthpro2-l"), new VisionRequest { Image = Image(), Mode = VisionMode.MetricDepth, FocalLengthPx = 300 }));
        Assert.Contains("depthpro2-l", ex.Message);
    }
}
