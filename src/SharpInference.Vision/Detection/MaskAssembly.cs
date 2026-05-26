using SharpInference.Core.Tensors;

namespace SharpInference.Vision.Detection;

/// <summary>Combines per-detection mask coefficients with the prototype tensor to produce a
/// binary instance mask per detection. Mirrors the post-NMS mask assembly used by Ultralytics:
/// <list type="number">
///   <item><b>Linear combine</b>: <c>mask = coefficients @ prototypes_flat</c> (matrix-vector product, where prototypes flatten the <c>[nm, protoH, protoW]</c> tensor to <c>[nm, protoH·protoW]</c>).</item>
///   <item><b>Sigmoid</b>: pointwise <c>1/(1+exp(-x))</c>.</item>
///   <item><b>Letterbox un-projection</b>: crop the proto-resolution mask to the slice that
///         corresponds to the original image (removing letterbox padding from the proto-space).</item>
///   <item><b>Bbox crop</b>: zero out pixels outside the detection's bounding box (Ultralytics'
///         <c>crop_mask</c> step — prevents one detection's coefficients from "bleeding" through
///         the entire image).</item>
///   <item><b>Bilinear resize</b>: upsample to the original-image resolution. (Optional — callers
///         can request the proto-resolution mask if they want to do their own resampling.)</item>
///   <item><b>Threshold</b>: binarize at 0.5 (Ultralytics default).</item>
/// </list>
/// <para>The combine step is small enough (32 × 25600 = ~820K multiplies per detection at the
/// canonical proto resolution) to run as a straight CPU loop without needing BLAS. Total work
/// for ~5 detections on a 640² image is well under 5 ms on a modern desktop.</para></summary>
public static unsafe class MaskAssembly
{
    /// <summary>Assembles one binary mask in original-image coordinates for the given detection.
    /// Returns an HxW grid of 0/1 bytes (row-major).</summary>
    /// <param name="detection">Detection with xyxy box in source coords (post-Transform.Invert).</param>
    /// <param name="coefficients">Per-detection mask coefficients, length <c>nm</c>.</param>
    /// <param name="prototypes">Proto output tensor <c>[1, nm, protoH, protoW]</c>.</param>
    /// <param name="transform">The letterbox transform from preprocessing.</param>
    /// <param name="threshold">Sigmoid threshold for binarization (default 0.5).</param>
    public static byte[] AssembleMask(
        YoloDetection detection,
        ReadOnlySpan<float> coefficients,
        Tensor prototypes,
        YoloPreprocessor.Transform transform,
        float threshold = 0.5f)
    {
        ArgumentNullException.ThrowIfNull(prototypes);
        if (prototypes.Shape.Rank != 4 || prototypes.Shape[0] != 1)
            throw new ArgumentException($"prototypes must be [1, nm, protoH, protoW]; got {prototypes.Shape}.", nameof(prototypes));

        int nm = (int)prototypes.Shape[1];
        int protoH = (int)prototypes.Shape[2];
        int protoW = (int)prototypes.Shape[3];
        if (coefficients.Length != nm)
            throw new ArgumentException($"coefficients length {coefficients.Length} != nm {nm}.", nameof(coefficients));

        // Step 1+2: linear combine + sigmoid into a proto-resolution mask buffer.
        // Mask values are stored as F32 for the intermediate (we threshold at the very end).
        float[] protoMask = new float[protoH * protoW];
        float* protoPtr = (float*)prototypes.DataPointer;
        for (int k = 0; k < nm; k++)
        {
            float c = coefficients[k];
            if (c == 0f) continue;
            float* layer = protoPtr + (long)k * protoH * protoW;
            for (int i = 0; i < protoMask.Length; i++)
                protoMask[i] += c * layer[i];
        }
        for (int i = 0; i < protoMask.Length; i++)
            protoMask[i] = 1f / (1f + MathF.Exp(-protoMask[i]));

        // Step 3: figure out the proto-space ROI that corresponds to the actual image content
        // (i.e. strip the letterbox padding in proto coordinates). Proto resolution is
        // (letterboxH / 4, letterboxW / 4) because the proto upsamples P3 (stride 8) by 2 → stride 4.
        float protoScaleX = (float)protoW / transform.PaddedWidth;
        float protoScaleY = (float)protoH / transform.PaddedHeight;
        int protoPadLeft = (int)MathF.Floor(transform.PadLeft * protoScaleX);
        int protoPadTop = (int)MathF.Floor(transform.PadTop * protoScaleY);
        int protoImgW = protoW - 2 * protoPadLeft; // assumes symmetric padding (true for our preprocessor)
        int protoImgH = protoH - 2 * protoPadTop;
        // Some letterbox configs have odd padding; in that case adjust the ROI directly using
        // resizedW/H instead of the assumed symmetric padding.
        protoImgW = (int)MathF.Round(transform.ResizedWidth * protoScaleX);
        protoImgH = (int)MathF.Round(transform.ResizedHeight * protoScaleY);

        // Resize the proto-space ROI to the source-image size via bilinear interpolation.
        // We also crop to the detection's bbox during the resize loop to save work.
        int srcW = transform.SourceWidth;
        int srcH = transform.SourceHeight;
        byte[] outMask = new byte[srcW * srcH];

        // Clamp the detection's bbox in source coords to image bounds.
        int x1 = Math.Clamp((int)MathF.Floor(detection.X1), 0, srcW);
        int y1 = Math.Clamp((int)MathF.Floor(detection.Y1), 0, srcH);
        int x2 = Math.Clamp((int)MathF.Ceiling(detection.X2), 0, srcW);
        int y2 = Math.Clamp((int)MathF.Ceiling(detection.Y2), 0, srcH);

        // Forward map from source pixel (xs, ys) to proto coord:
        //   px = (xs / srcW) * protoImgW + protoPadLeft
        //   py = (ys / srcH) * protoImgH + protoPadTop
        // We then bilinear-sample protoMask at (px, py) and threshold.
        float scaleProtoX = (float)protoImgW / srcW;
        float scaleProtoY = (float)protoImgH / srcH;

        for (int ys = y1; ys < y2; ys++)
        {
            float pyf = (ys + 0.5f) * scaleProtoY + protoPadTop - 0.5f;
            int py0 = (int)MathF.Floor(pyf);
            float fy = pyf - py0;
            int py0c = Math.Clamp(py0, 0, protoH - 1);
            int py1c = Math.Clamp(py0 + 1, 0, protoH - 1);
            float wy1 = fy, wy0 = 1f - fy;

            for (int xs = x1; xs < x2; xs++)
            {
                float pxf = (xs + 0.5f) * scaleProtoX + protoPadLeft - 0.5f;
                int px0 = (int)MathF.Floor(pxf);
                float fx = pxf - px0;
                int px0c = Math.Clamp(px0, 0, protoW - 1);
                int px1c = Math.Clamp(px0 + 1, 0, protoW - 1);
                float wx1 = fx, wx0 = 1f - fx;

                float v00 = protoMask[py0c * protoW + px0c];
                float v01 = protoMask[py0c * protoW + px1c];
                float v10 = protoMask[py1c * protoW + px0c];
                float v11 = protoMask[py1c * protoW + px1c];
                float v = (v00 * wx0 + v01 * wx1) * wy0 + (v10 * wx0 + v11 * wx1) * wy1;
                outMask[ys * srcW + xs] = v >= threshold ? (byte)1 : (byte)0;
            }
        }

        return outMask;
    }
}
