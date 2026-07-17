namespace HartsyInference.Vision.FaceDetection;

/// <summary>The five facial landmarks a YOLOv8-Face detector emits per face, in the WIDERFACE / RetinaFace
/// ordering — the same order as the ArcFace alignment template, so <see cref="ToXyArray"/> feeds straight into
/// <see cref="HartsyInference.Vision.Face.FaceAlignment.AlignToTemplate"/> without reindexing.
/// <para>Point semantics (left/right are image-side, matched to the ArcFace template rows): 0 left-eye,
/// 1 right-eye, 2 nose, 3 mouth-left, 4 mouth-right. The names are the convention a standard face checkpoint
/// uses; a checkpoint trained with a different keypoint order needs its points remapped before construction.</para></summary>
public readonly struct Landmark5
{
    /// <summary>Number of landmark points.</summary>
    public const int Count = 5;

    public FaceLandmark LeftEye { get; }
    public FaceLandmark RightEye { get; }
    public FaceLandmark Nose { get; }
    public FaceLandmark MouthLeft { get; }
    public FaceLandmark MouthRight { get; }

    public Landmark5(FaceLandmark leftEye, FaceLandmark rightEye, FaceLandmark nose, FaceLandmark mouthLeft, FaceLandmark mouthRight)
    {
        LeftEye = leftEye;
        RightEye = rightEye;
        Nose = nose;
        MouthLeft = mouthLeft;
        MouthRight = mouthRight;
    }

    /// <summary>Builds from a flat (x, y[, score]) sequence of exactly <see cref="Count"/> points. <paramref name="stride"/>
    /// is the values-per-point (2 for x/y, 3 for x/y/score) — the layout the detector's landmark branch produces.</summary>
    public static Landmark5 FromFlat(ReadOnlySpan<float> flat, int stride)
    {
        if (stride != 2 && stride != 3)
            throw new ArgumentOutOfRangeException(nameof(stride), stride, "Landmark stride must be 2 (x,y) or 3 (x,y,score).");
        if (flat.Length != Count * stride)
            throw new ArgumentException($"Expected {Count * stride} floats for {Count} points at stride {stride}; got {flat.Length}.", nameof(flat));
        Span<FaceLandmark> pts = stackalloc FaceLandmark[Count];
        for (int i = 0; i < Count; i++)
        {
            float score = stride == 3 ? flat[i * stride + 2] : 1f;
            pts[i] = new FaceLandmark(flat[i * stride + 0], flat[i * stride + 1], score);
        }
        return new Landmark5(pts[0], pts[1], pts[2], pts[3], pts[4]);
    }

    /// <summary>Point by index in canonical order (0 left-eye … 4 mouth-right).</summary>
    public FaceLandmark this[int index] => index switch
    {
        0 => LeftEye,
        1 => RightEye,
        2 => Nose,
        3 => MouthLeft,
        4 => MouthRight,
        _ => throw new ArgumentOutOfRangeException(nameof(index), index, "Landmark index must be 0..4."),
    };

    /// <summary>Flattens to <c>[x0, y0, x1, y1, … x4, y4]</c> (length 10) in template order — the direct input to
    /// the ArcFace similarity fit.</summary>
    public float[] ToXyArray()
    {
        float[] xy = new float[Count * 2];
        for (int i = 0; i < Count; i++)
        {
            FaceLandmark p = this[i];
            xy[i * 2 + 0] = p.X;
            xy[i * 2 + 1] = p.Y;
        }
        return xy;
    }
}
