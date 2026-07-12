namespace HartsyInference.Vision.Detection;

/// <summary>YOLO architecture configuration — variant-specific scaling parameters that drive
/// channel counts and bottleneck repeat counts. Presets cover the YOLOv8 n/s/m/l/x family;
/// YOLO11 support is deferred (it adds C3k2 + C2PSA blocks).</summary>
public sealed record YoloConfig
{
    /// <summary>Variant identifier (e.g. <c>"yolov8n"</c>) — used for logging and asset lookup, not for code paths.</summary>
    public required string Name { get; init; }

    /// <summary>Multiplier applied to base bottleneck repeat counts (the depth side of compound scaling).</summary>
    public required float DepthMultiple { get; init; }

    /// <summary>Multiplier applied to base channel counts (the width side of compound scaling).</summary>
    public required float WidthMultiple { get; init; }

    /// <summary>Channel cap applied after multiplication, before rounding to the nearest multiple of 8.</summary>
    public required int MaxChannels { get; init; }

    /// <summary>Number of detection classes (80 for COCO).</summary>
    public required int NumClasses { get; init; }

    /// <summary>DFL distribution bin count (16 for YOLOv8).</summary>
    public int RegMax { get; init; } = 16;

    /// <summary>Keypoints per detection for pose models (17 for COCO body pose). 0 = a plain detector (no keypoint head).</summary>
    public int NumKeypoints { get; init; } = 0;

    /// <summary>Values per keypoint (3 = x, y, visibility). Ignored when <see cref="NumKeypoints"/> is 0.</summary>
    public int KptDims { get; init; } = 3;

    /// <summary>Per-detect-scale strides relative to input. Always <c>[8, 16, 32]</c> for YOLOv8.</summary>
    public IReadOnlyList<int> Strides { get; init; } = [8, 16, 32];

    /// <summary>YOLOv8 base channels at the five backbone stages (P1/2, P2/4, P3/8, P4/16, P5/32). Unscaled.</summary>
    public IReadOnlyList<int> BaseChannels { get; init; } = [64, 128, 256, 512, 1024];

    /// <summary>YOLOv8 base C2f repeat counts for the four C2f stages in the backbone. Unscaled.</summary>
    public IReadOnlyList<int> BaseBackboneRepeats { get; init; } = [3, 6, 6, 3];

    /// <summary>YOLOv8 base C2f repeat count for each neck C2f block (4 of them). Unscaled.</summary>
    public IReadOnlyList<int> BaseNeckRepeats { get; init; } = [3, 3, 3, 3];

    /// <summary>YOLOv8n — smallest variant (3.2M params, 37.3 mAP@0.5:0.95).</summary>
    public static YoloConfig YoloV8n => new()
    {
        Name = "yolov8n",
        DepthMultiple = 0.33f,
        WidthMultiple = 0.25f,
        MaxChannels = 1024,
        NumClasses = 80,
    };

    /// <summary>YOLOv8s (11.2M params, 44.9 mAP).</summary>
    public static YoloConfig YoloV8s => YoloV8n with { Name = "yolov8s", WidthMultiple = 0.50f };

    /// <summary>YOLOv8m (25.9M params, 50.2 mAP).</summary>
    public static YoloConfig YoloV8m => new()
    {
        Name = "yolov8m",
        DepthMultiple = 0.67f,
        WidthMultiple = 0.75f,
        MaxChannels = 768,
        NumClasses = 80,
    };

    /// <summary>YOLOv8l (43.7M params, 52.9 mAP).</summary>
    public static YoloConfig YoloV8l => new()
    {
        Name = "yolov8l",
        DepthMultiple = 1.00f,
        WidthMultiple = 1.00f,
        MaxChannels = 512,
        NumClasses = 80,
    };

    /// <summary>YOLOv8x (68.2M params, 53.9 mAP).</summary>
    public static YoloConfig YoloV8x => YoloV8l with { Name = "yolov8x", WidthMultiple = 1.25f };

    // ── YOLO11 presets ────────────────────────────────────────────────────────
    //
    // YOLO11 architecture has a different layer layout than v8 (extra C2PSA + reshuffled C3k2
    // channel progression); the channel-resolution helpers in this record happen to work for
    // both because the scaling formula <c>make_divisible(min(c_base, max_channels) * width, 8)</c>
    // is identical. The YAML layer counts diverge between the two families, so the model class
    // (YoloModel vs YoloV11Model) is the architecture switch — these config presets are only
    // the scaling knobs.

    /// <summary>YOLO11n — smallest variant (2.6M params, 39.5 mAP@0.5:0.95 — ~19% fewer params than v8n with +2.2 mAP).</summary>
    public static YoloConfig YoloV11n => new()
    {
        Name = "yolo11n",
        DepthMultiple = 0.50f,
        WidthMultiple = 0.25f,
        MaxChannels = 1024,
        NumClasses = 80,
    };

    /// <summary>YOLO11s (9.4M params, 47.0 mAP).</summary>
    public static YoloConfig YoloV11s => YoloV11n with { Name = "yolo11s", WidthMultiple = 0.50f };

    /// <summary>YOLO11m (20.1M params, 51.5 mAP).</summary>
    public static YoloConfig YoloV11m => YoloV11n with { Name = "yolo11m", WidthMultiple = 1.00f, MaxChannels = 512 };

    /// <summary>YOLO11l (25.3M params, 53.4 mAP).</summary>
    public static YoloConfig YoloV11l => new()
    {
        Name = "yolo11l",
        DepthMultiple = 1.00f,
        WidthMultiple = 1.00f,
        MaxChannels = 512,
        NumClasses = 80,
    };

    /// <summary>YOLO11x (56.9M params, 54.7 mAP).</summary>
    public static YoloConfig YoloV11x => YoloV11l with { Name = "yolo11x", WidthMultiple = 1.50f };

    /// <summary>YOLO11n-pose — the nano pose variant: 1 class (person) + a 17-keypoint (COCO body) head.
    /// Same backbone/neck as <see cref="YoloV11n"/>; the detect head gains a cv4 keypoint branch.</summary>
    public static YoloConfig YoloV11nPose => new()
    {
        Name = "yolo11n-pose",
        DepthMultiple = 0.50f,
        WidthMultiple = 0.25f,
        MaxChannels = 1024,
        NumClasses = 1,
        NumKeypoints = 17,
        KptDims = 3,
    };

    /// <summary>Resolved per-stage backbone channels matching Ultralytics' formula
    /// <c>make_divisible(min(c_base, max_channels) * width, 8)</c> — i.e. cap to <c>max_channels</c>
    /// <i>first</i>, then multiply by width, then ceil-round to a multiple of 8. Indexed 0..4
    /// across the five backbone stages (P1/2 → P5/32).</summary>
    public int ScaledChannel(int stageIndex)
    {
        int baseCh = BaseChannels[stageIndex];
        int capped = Math.Min(baseCh, MaxChannels);
        return MakeDivisible((int)MathF.Round(capped * WidthMultiple), 8);
    }

    /// <summary>Resolved per-stage C2f repeat count after applying depth multiplier.
    /// Index 0..3 across the four C2f stages in the backbone.</summary>
    public int ScaledBackboneRepeat(int stageIndex)
    {
        int baseRepeat = BaseBackboneRepeats[stageIndex];
        int scaled = Math.Max((int)MathF.Round(baseRepeat * DepthMultiple), 1);
        return scaled;
    }

    /// <summary>Resolved per-stage C2f repeat for the neck. Index 0..3 across the four C2f blocks.</summary>
    public int ScaledNeckRepeat(int stageIndex)
    {
        int baseRepeat = BaseNeckRepeats[stageIndex];
        return Math.Max((int)MathF.Round(baseRepeat * DepthMultiple), 1);
    }

    private static int MakeDivisible(int v, int divisor)
    {
        return ((v + divisor - 1) / divisor) * divisor;
    }
}
