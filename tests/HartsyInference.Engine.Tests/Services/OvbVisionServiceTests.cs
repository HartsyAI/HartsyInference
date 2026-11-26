using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The Open Vision Banana routing in VisionService, without weights: which modes reach the model, which are refused by
/// name, and the class-list rule for segmentation.</summary>
public sealed class OvbVisionServiceTests
{
    private static ImageData Image() => new() { Rgb = new byte[8 * 6 * 3], Width = 8, Height = 6 };

    private static ModelSpec Spec(string id) =>
        new() { Requested = id, Modality = Modality.Vision, Catalog = ModelCatalog.Find(id), LocalPath = null };

    [Theory]
    [InlineData("open-vision-banana-4b")]
    [InlineData("open-vision-banana-9b")]
    public async Task SegmentWithNoClassNamesIsRefusedBeforeAnyLoad(string id)
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        ArgumentException ex = await Assert.ThrowsAsync<ArgumentException>(() =>
            engine.Vision.RunAsync(Spec(id), new VisionRequest { Image = Image(), Mode = VisionMode.Segment, Prompt = " , " }));
        Assert.Contains("comma-separated", ex.Message);
    }

    [Theory]
    [InlineData(VisionMode.Detect)]
    [InlineData(VisionMode.Embed)]
    [InlineData(VisionMode.Edge)]
    [InlineData(VisionMode.Matting)]
    public async Task ModesTheModelDoesNotAnswerAreRefusedByName(VisionMode mode)
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        NotSupportedException ex = await Assert.ThrowsAsync<NotSupportedException>(() =>
            engine.Vision.RunAsync(Spec("open-vision-banana-4b"), new VisionRequest { Image = Image(), Mode = mode, Prompt = "car" }));
        Assert.Contains("open-vision-banana-4b", ex.Message);
        Assert.Contains(mode.ToString(), ex.Message);
    }

    [Theory]
    [InlineData(VisionMode.Segment)]
    [InlineData(VisionMode.Depth)]
    [InlineData(VisionMode.Normal)]
    public async Task SegmentDepthAndNormalsReachTheModelAndFailOnlyAtLoad(VisionMode mode)
    {
        // The routing hands these to the model. With no checkpoint the failure comes from the load, not from the mode switch.
        using InferenceEngine engine = new InferenceEngine("cpu");
        Exception ex = await Assert.ThrowsAnyAsync<Exception>(() =>
            engine.Vision.RunAsync(Spec("open-vision-banana-4b"), new VisionRequest { Image = Image(), Mode = mode, Prompt = "car" }));
        Assert.IsNotType<NotSupportedException>(ex);
        Assert.IsNotType<ArgumentException>(ex);
    }
}
