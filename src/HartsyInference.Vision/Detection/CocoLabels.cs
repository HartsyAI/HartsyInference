namespace HartsyInference.Vision.Detection;

/// <summary>COCO 80-class label vocabulary used by every Ultralytics YOLO checkpoint trained on
/// MS-COCO. Indices match the order in <c>ultralytics/cfg/datasets/coco.yaml</c> so passing a
/// YOLO class-id directly into <see cref="Get"/> returns the right label.
/// <para>If a checkpoint was trained on a different dataset (OpenImages, custom classes), supply
/// your own label table — the COCO mapping is only correct for COCO-pretrained models.</para></summary>
public static class CocoLabels
{
    /// <summary>The 80 COCO class names in canonical order. Read-only; do not mutate.</summary>
    public static readonly IReadOnlyList<string> Names = [
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    ];

    /// <summary>Total class count (80).</summary>
    public const int Count = 80;

    /// <summary>Returns the label for a class index, or <c>"class_{id}"</c> for indices outside the COCO range. Never throws — useful when a model emits a class index that doesn't match the assumed vocab (e.g. a fine-tuned head with extra classes).</summary>
    public static string Get(int classId)
    {
        if (classId < 0 || classId >= Names.Count)
            return $"class_{classId}";
        return Names[classId];
    }
}
