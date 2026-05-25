using SharpInference.Vision.Detection;
using Xunit;

namespace SharpInference.Vision.Tests;

/// <summary>Tests for the YOLO NMS algorithm.</summary>
public sealed class NonMaxSuppressionTests
{
    [Fact]
    public void Run_EmptyInput_ReturnsEmpty()
    {
        IReadOnlyList<YoloDetection> result = NonMaxSuppression.Run([]);
        Assert.Empty(result);
    }

    [Fact]
    public void Run_TwoOverlappingSameClass_KeepsTopScored()
    {
        // Two near-identical boxes, same class. The higher-scored one wins.
        YoloDetection a = new(10, 10, 100, 100, 0.9f, ClassId: 0);
        YoloDetection b = new(12, 12, 102, 102, 0.6f, ClassId: 0);
        IReadOnlyList<YoloDetection> result = NonMaxSuppression.Run([a, b], iouThreshold: 0.5f);
        Assert.Single(result);
        Assert.Equal(0.9f, result[0].Confidence);
    }

    [Fact]
    public void Run_TwoOverlappingDifferentClass_KeepsBoth()
    {
        // Same boxes, different classes — class-specific NMS keeps both.
        YoloDetection a = new(10, 10, 100, 100, 0.9f, ClassId: 0);
        YoloDetection b = new(12, 12, 102, 102, 0.6f, ClassId: 1);
        IReadOnlyList<YoloDetection> result = NonMaxSuppression.Run([a, b], iouThreshold: 0.5f);
        Assert.Equal(2, result.Count);
    }

    [Fact]
    public void Run_TwoOverlappingDifferentClass_ClassAgnostic_KeepsOnlyTop()
    {
        YoloDetection a = new(10, 10, 100, 100, 0.9f, ClassId: 0);
        YoloDetection b = new(12, 12, 102, 102, 0.6f, ClassId: 1);
        IReadOnlyList<YoloDetection> result = NonMaxSuppression.Run([a, b], iouThreshold: 0.5f, classAgnostic: true);
        Assert.Single(result);
        Assert.Equal(0, result[0].ClassId);
    }

    [Fact]
    public void Run_DisjointBoxes_AllSurvive()
    {
        YoloDetection a = new(0, 0, 50, 50, 0.9f, ClassId: 0);
        YoloDetection b = new(100, 100, 150, 150, 0.8f, ClassId: 0);
        YoloDetection c = new(200, 200, 250, 250, 0.7f, ClassId: 0);
        IReadOnlyList<YoloDetection> result = NonMaxSuppression.Run([a, b, c]);
        Assert.Equal(3, result.Count);
    }

    [Fact]
    public void Run_HonorsMaxDetectionsCap()
    {
        List<YoloDetection> input = [];
        // 10 disjoint boxes — none suppress each other; expect cap.
        for (int i = 0; i < 10; i++)
            input.Add(new(i * 100, 0, i * 100 + 50, 50, 0.5f + i * 0.01f, ClassId: 0));
        IReadOnlyList<YoloDetection> result = NonMaxSuppression.Run(input, maxDetections: 3);
        Assert.Equal(3, result.Count);
        // Sorted by confidence desc — top three are i=9, 8, 7.
        Assert.Equal(0.59f, result[0].Confidence, tolerance: 1e-5f);
        Assert.Equal(0.58f, result[1].Confidence, tolerance: 1e-5f);
        Assert.Equal(0.57f, result[2].Confidence, tolerance: 1e-5f);
    }

    [Fact]
    public void Run_SortsByConfidenceDescending()
    {
        // Input is reverse-sorted; NMS must still produce highest-first output.
        YoloDetection low = new(0, 0, 50, 50, 0.3f, ClassId: 0);
        YoloDetection mid = new(200, 200, 250, 250, 0.6f, ClassId: 0);
        YoloDetection high = new(400, 400, 450, 450, 0.9f, ClassId: 0);
        IReadOnlyList<YoloDetection> result = NonMaxSuppression.Run([low, mid, high]);
        Assert.Equal(3, result.Count);
        Assert.Equal(0.9f, result[0].Confidence);
        Assert.Equal(0.6f, result[1].Confidence);
        Assert.Equal(0.3f, result[2].Confidence);
    }

    [Fact]
    public void RunWithConfidenceFilter_DropsBelowThreshold()
    {
        YoloDetection keep = new(0, 0, 50, 50, 0.8f, ClassId: 0);
        YoloDetection drop = new(200, 200, 250, 250, 0.1f, ClassId: 0);
        IReadOnlyList<YoloDetection> result = NonMaxSuppression.RunWithConfidenceFilter(
            [keep, drop], confidenceThreshold: 0.5f);
        Assert.Single(result);
        Assert.Equal(0.8f, result[0].Confidence);
    }

    [Fact]
    public void Iou_IdenticalBoxes_IsOne()
    {
        YoloDetection a = new(10, 20, 100, 200, 0.5f, ClassId: 0);
        Assert.Equal(1f, a.Iou(a), tolerance: 1e-6f);
    }

    [Fact]
    public void Iou_DisjointBoxes_IsZero()
    {
        YoloDetection a = new(0, 0, 50, 50, 0.5f, ClassId: 0);
        YoloDetection b = new(100, 100, 150, 150, 0.5f, ClassId: 0);
        Assert.Equal(0f, a.Iou(b));
    }

    [Fact]
    public void Iou_FullyContainedBox_ProducesAreaRatio()
    {
        // 10×10 box fully inside 20×20 box — intersection = 100, union = 400.
        YoloDetection outer = new(0, 0, 20, 20, 0.5f, ClassId: 0);
        YoloDetection inner = new(5, 5, 15, 15, 0.5f, ClassId: 0);
        Assert.Equal(0.25f, outer.Iou(inner), tolerance: 1e-6f);
    }

    [Fact]
    public void Run_RejectsInvalidIouThreshold()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => NonMaxSuppression.Run([], iouThreshold: -0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => NonMaxSuppression.Run([], iouThreshold: 1.5f));
    }

    [Fact]
    public void CocoLabels_GetsKnownClasses()
    {
        Assert.Equal("person", CocoLabels.Get(0));
        Assert.Equal("cat", CocoLabels.Get(15));
        Assert.Equal("dog", CocoLabels.Get(16));
        Assert.Equal("toothbrush", CocoLabels.Get(79));
        Assert.Equal(80, CocoLabels.Count);
    }

    [Fact]
    public void CocoLabels_OutOfRange_ReturnsFallback()
    {
        Assert.Equal("class_-1", CocoLabels.Get(-1));
        Assert.Equal("class_999", CocoLabels.Get(999));
    }
}
