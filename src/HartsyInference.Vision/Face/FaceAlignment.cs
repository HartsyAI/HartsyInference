using HartsyInference.Vision.Detection;

namespace HartsyInference.Vision.Face;

/// <summary>ArcFace-standard face alignment: estimates a least-squares similarity transform from detected facial
/// keypoints to the canonical ArcFace 112×112 template, then warps the source image into an aligned crop
/// (equivalent to insightface's <c>norm_crop</c> / skimage <c>SimilarityTransform.estimate</c> + <c>cv2.warpAffine</c>).
///
/// <para><b>Deviation from insightface/SCRFD:</b> SCRFD supplies 5 landmarks (eyes, nose, mouth corners); the
/// engine's YOLO11-pose detector provides COCO-17 keypoints, which include eyes and nose but no mouth. The
/// similarity transform (4 DOF) is therefore estimated from the 3 available correspondences (both eyes + nose)
/// — still over-determined (6 equations) and identical in the exact-fit case, but slightly less robust to
/// individual keypoint noise than the 5-point fit. Empirically the resulting crops are close enough that
/// ArcFace embeddings of the same identity stay highly similar.</para></summary>
public static class FaceAlignment
{
    /// <summary>Aligned crop side in pixels (ArcFace standard).</summary>
    public const int CropSize = 112;

    /// <summary>Canonical ArcFace 112×112 landmark template (insightface <c>arcface_dst</c>), image-left eye,
    /// image-right eye, nose, mouth-left, mouth-right.</summary>
    public static readonly float[,] Template =
    {
        { 38.2946f, 51.6963f },
        { 73.5318f, 51.5014f },
        { 56.0252f, 71.7366f },
        { 41.5493f, 92.3655f },
        { 70.7299f, 92.2041f },
    };

    /// <summary>A 2×3 row-major affine matrix <c>[a b tx; c d ty]</c> mapping source pixels to destination pixels.</summary>
    public readonly record struct Affine2x3(float A, float B, float Tx, float C, float D, float Ty)
    {
        /// <summary>Applies the transform to a point.</summary>
        public (float X, float Y) Apply(float x, float y) => (A * x + B * y + Tx, C * x + D * y + Ty);

        /// <summary>Inverts the affine transform. Throws when the linear part is singular.</summary>
        public Affine2x3 Invert()
        {
            float det = A * D - B * C;
            if (MathF.Abs(det) < 1e-12f)
                throw new InvalidOperationException("Cannot invert a degenerate similarity transform (collinear or coincident keypoints?).");
            float ia = D / det, ib = -B / det, ic = -C / det, id = A / det;
            return new Affine2x3(ia, ib, -(ia * Tx + ib * Ty), ic, id, -(ic * Tx + id * Ty));
        }
    }

    /// <summary>Least-squares non-reflective similarity transform (scale + rotation + translation) mapping
    /// <paramref name="src"/> points onto <paramref name="dst"/> points. Requires ≥ 2 non-coincident points.
    /// This is the closed-form Umeyama solution restricted to positive determinant — identical to skimage's
    /// <c>SimilarityTransform.estimate</c> for non-degenerate face landmarks.</summary>
    public static Affine2x3 EstimateSimilarity(ReadOnlySpan<float> src, ReadOnlySpan<float> dst)
    {
        if (src.Length != dst.Length || src.Length < 4 || src.Length % 2 != 0)
            throw new ArgumentException($"Need matching flat (x,y) arrays of >= 2 points; got src={src.Length}, dst={dst.Length} floats.");
        int n = src.Length / 2;

        double srcMeanX = 0, srcMeanY = 0, dstMeanX = 0, dstMeanY = 0;
        for (int i = 0; i < n; i++)
        {
            srcMeanX += src[2 * i]; srcMeanY += src[2 * i + 1];
            dstMeanX += dst[2 * i]; dstMeanY += dst[2 * i + 1];
        }
        srcMeanX /= n; srcMeanY /= n; dstMeanX /= n; dstMeanY /= n;

        // Solve min Σ ||s·R·p_i + t − q_i||² over centered points:
        //   a = Σ(p'·q') / Σ||p'||²   (dot terms)
        //   b = Σ(p'×q') / Σ||p'||²   (cross terms)
        double dot = 0, cross = 0, srcVar = 0;
        for (int i = 0; i < n; i++)
        {
            double px = src[2 * i] - srcMeanX, py = src[2 * i + 1] - srcMeanY;
            double qx = dst[2 * i] - dstMeanX, qy = dst[2 * i + 1] - dstMeanY;
            dot += px * qx + py * qy;
            cross += px * qy - py * qx;
            srcVar += px * px + py * py;
        }
        if (srcVar < 1e-9)
            throw new ArgumentException("Source keypoints are (nearly) coincident — cannot estimate a similarity transform.");
        double a = dot / srcVar;
        double b = cross / srcVar;
        double tx = dstMeanX - (a * srcMeanX - b * srcMeanY);
        double ty = dstMeanY - (b * srcMeanX + a * srcMeanY);
        return new Affine2x3((float)a, (float)-b, (float)tx, (float)b, (float)a, (float)ty);
    }

    /// <summary>Extracts the 3 alignment keypoints (image-left eye, image-right eye, nose) from a YOLO pose
    /// detection as a flat (x,y) array matched to <see cref="Template"/> rows 0–2. The two COCO eye keypoints
    /// are ordered by image x so the mapping holds regardless of the detector's left/right semantics. Returns
    /// false when any of the three points is below <paramref name="visThreshold"/>.</summary>
    public static bool TryGetAlignmentPoints(PoseDetection person, float visThreshold, out float[] points)
    {
        points = [];
        if (person.Keypoints.Count < 3) return false;
        Keypoint nose = person.Keypoints[0];
        Keypoint eyeA = person.Keypoints[1];
        Keypoint eyeB = person.Keypoints[2];
        if (nose.Confidence < visThreshold || eyeA.Confidence < visThreshold || eyeB.Confidence < visThreshold)
            return false;
        Keypoint leftMost = eyeA.X <= eyeB.X ? eyeA : eyeB;
        Keypoint rightMost = eyeA.X <= eyeB.X ? eyeB : eyeA;
        points = [leftMost.X, leftMost.Y, rightMost.X, rightMost.Y, nose.X, nose.Y];
        return true;
    }

    /// <summary>Estimates the similarity from detected keypoints to the first <c>points.Length/2</c> rows of
    /// the ArcFace template and warps the source image into an <paramref name="outputSize"/>² aligned RGB crop
    /// (bilinear sampling, black border). <paramref name="rgbPixels"/> is HWC-packed RGB u8. The template is
    /// scaled by <c>outputSize / 112</c>, matching insightface <c>norm_crop(image_size=…)</c> for multiples of
    /// 112 — FaceID-Plus feeds the same alignment at 224 to CLIP-Vision.</summary>
    public static byte[] AlignToTemplate(ReadOnlySpan<byte> rgbPixels, int width, int height, ReadOnlySpan<float> points, int outputSize = CropSize)
    {
        int n = points.Length / 2;
        if (n < 2 || n > 5)
            throw new ArgumentException($"Expected 2–5 (x,y) keypoints matching the template rows; got {n}.");
        if (outputSize <= 0 || outputSize % CropSize != 0)
            throw new ArgumentException($"outputSize must be a positive multiple of {CropSize} (insightface norm_crop contract); got {outputSize}.", nameof(outputSize));
        float ratio = outputSize / (float)CropSize;
        Span<float> dst = stackalloc float[n * 2];
        for (int i = 0; i < n; i++)
        {
            dst[2 * i] = Template[i, 0] * ratio;
            dst[2 * i + 1] = Template[i, 1] * ratio;
        }
        Affine2x3 srcToDst = EstimateSimilarity(points, dst);
        return WarpAffine(rgbPixels, width, height, srcToDst, outputSize, outputSize);
    }

    /// <summary>Warps the source image with <paramref name="srcToDst"/> into a <paramref name="outWidth"/>×
    /// <paramref name="outHeight"/> RGB crop: each destination pixel bilinearly samples the source at the
    /// inverse-mapped position (out-of-bounds → black), matching <c>cv2.warpAffine</c> defaults.</summary>
    public static byte[] WarpAffine(ReadOnlySpan<byte> rgbPixels, int width, int height, Affine2x3 srcToDst, int outWidth, int outHeight)
    {
        if (rgbPixels.Length != (long)width * height * 3)
            throw new ArgumentException($"rgbPixels length {rgbPixels.Length} != width*height*3 ({(long)width * height * 3}).");
        Affine2x3 dstToSrc = srcToDst.Invert();
        byte[] output = new byte[outWidth * outHeight * 3];
        for (int oy = 0; oy < outHeight; oy++)
        {
            for (int ox = 0; ox < outWidth; ox++)
            {
                (float sx, float sy) = dstToSrc.Apply(ox, oy);
                int x0 = (int)MathF.Floor(sx);
                int y0 = (int)MathF.Floor(sy);
                float fx = sx - x0;
                float fy = sy - y0;
                int outOff = (oy * outWidth + ox) * 3;
                for (int c = 0; c < 3; c++)
                {
                    float v00 = SampleOrZero(rgbPixels, width, height, x0, y0, c);
                    float v10 = SampleOrZero(rgbPixels, width, height, x0 + 1, y0, c);
                    float v01 = SampleOrZero(rgbPixels, width, height, x0, y0 + 1, c);
                    float v11 = SampleOrZero(rgbPixels, width, height, x0 + 1, y0 + 1, c);
                    float top = v00 + (v10 - v00) * fx;
                    float bottom = v01 + (v11 - v01) * fx;
                    float v = top + (bottom - top) * fy;
                    output[outOff + c] = (byte)Math.Clamp((int)MathF.Round(v), 0, 255);
                }
            }
        }
        return output;
    }

    private static float SampleOrZero(ReadOnlySpan<byte> rgb, int width, int height, int x, int y, int channel)
    {
        if (x < 0 || x >= width || y < 0 || y >= height) return 0f;
        return rgb[(y * width + x) * 3 + channel];
    }
}
