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
        Assert.Contains("only applies to MetricDepth",
            Assert.Throws<ArgumentException>(() => VisionRequestValidator.Validate(new VisionRequest { Image = Image(), Mode = VisionMode.Depth, FocalLengthPx = 500 })).Message);
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
    public async Task ASingleImageWithoutTheCheckpointNamesTheRow()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        FileNotFoundException ex = await Assert.ThrowsAsync<FileNotFoundException>(() =>
            engine.Vision.RunAsync(Spec("depthpro2-l"), new VisionRequest { Image = Image(), Mode = VisionMode.MetricDepth, FocalLengthPx = 300 }));
        Assert.Contains("depthpro2-l", ex.Message);
    }
}
