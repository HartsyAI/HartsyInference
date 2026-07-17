using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Vision.Detection;
using HartsyInference.Vision.FaceDetection;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Structural tests for the YOLOv8-Face head + decode path. The synthetic-weight forward gates the head's
/// channel contract (4 box + 1 class + 5·ndim landmark channels); the decode tests pin the un-letterbox math and NMS
/// deterministically; the real-weight test is env-gated and skips when the checkpoint is absent.</summary>
public sealed class FaceDetectorForwardTests
{
    private const int NumClasses = 1;
    private const int RegMax = 16;
    private const int NumPoints = 5;
    private const int KptDims = 3;

    /// <summary>Head forward over random synthetic weights: asserts the box/class tensor is <c>[1, 4+1, A]</c> and the
    /// landmark tensor is <c>[1, 5·ndim, A]</c> over the same anchor count. This is the channel guarantee the whole
    /// <see cref="YoloV8FaceModel"/> inherits (the backbone/neck only feed these three scales into the head).</summary>
    [Fact]
    [Trait("Category", "SyntheticSmoke")]
    public unsafe void FaceDetectHead_Forward_ProducesBoxClassAndLandmarkChannels()
    {
        int[] inChannels = [16, 32, 64];
        int[] strides = [8, 16, 32];
        (int h, int w)[] scales = [(8, 8), (4, 4), (2, 2)];
        int totalAnchors = scales.Sum(s => s.h * s.w);

        FaceDetectHead head = new(NumClasses, RegMax, NumPoints, KptDims, inChannels, strides);
        Dictionary<string, Tensor> weights = BuildHeadWeights(inChannels, prefix: "head");
        head.LoadWeights(weights, "head");

        using IBackend backend = new CpuBackend();
        Tensor[] feats = new Tensor[inChannels.Length];
        for (int s = 0; s < inChannels.Length; s++)
            feats[s] = Random4D(1, inChannels[s], scales[s].h, scales[s].w, seed: 7000 + s);

        (Tensor detections, Tensor landmarks) = head.Forward(backend, feats);
        try
        {
            Assert.Equal(3, detections.Shape.Rank);
            Assert.Equal(1, (int)detections.Shape[0]);
            Assert.Equal(4 + NumClasses, (int)detections.Shape[1]);
            Assert.Equal(totalAnchors, (int)detections.Shape[2]);

            Assert.Equal(3, landmarks.Shape.Rank);
            Assert.Equal(1, (int)landmarks.Shape[0]);
            Assert.Equal(NumPoints * KptDims, (int)landmarks.Shape[1]);
            Assert.Equal(totalAnchors, (int)landmarks.Shape[2]);

            // Finiteness — random weights must not produce NaN/Inf in the decode.
            ReadOnlySpan<float> lm = landmarks.AsReadOnlySpan<float>();
            for (int i = 0; i < lm.Length; i++)
                Assert.True(float.IsFinite(lm[i]), $"Non-finite landmark value at {i}: {lm[i]}");
        }
        finally
        {
            detections.Dispose();
            landmarks.Dispose();
            foreach (Tensor t in feats) t.Dispose();
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }

    /// <summary>Pure landmark-decode math: a raw branch offset at a known grid cell maps to the expected source pixel
    /// after the head's <c>(2·r + g)·stride</c> grid decode and the letterbox inversion. Uses a non-trivial transform
    /// (half-scale + left/top padding) so the inversion actually does something.</summary>
    [Fact]
    public void LandmarkDecode_UnLetterbox_MapsAnchorOffsetToSourcePixels()
    {
        // Source 200×100 letterboxed into a 128×128 canvas: scale = min(128/100, 128/200) = 0.64.
        YoloPreprocessor.Transform transform = new YoloPreprocessor(128, stride: 32, stridePadAlign: false)
            .ComputeTransform(srcWidth: 200, srcHeight: 100);

        // Grid cell (3, 2) at stride 16, raw offset (0.25, 0.5):
        //   canvasX = (0.25*2 + 3)*16 = 56 ; canvasY = (0.5*2 + 2)*16 = 48
        float rawX = 0.25f, rawY = 0.5f;
        int gx = 3, gy = 2;
        float stride = 16f;
        float canvasX = (rawX * 2f + gx) * stride;
        float canvasY = (rawY * 2f + gy) * stride;
        (float expX, float expY) = transform.InvertPoint(canvasX, canvasY);

        (float x, float y) = LandmarkExtractor.DecodeRawPoint(rawX, rawY, gx, gy, stride, transform);

        Assert.Equal(expX, x, tolerance: 1e-4f);
        Assert.Equal(expY, y, tolerance: 1e-4f);
        // Sanity: within source bounds.
        Assert.InRange(x, 0f, 200f);
        Assert.InRange(y, 0f, 100f);
    }

    /// <summary>Decode over hand-built detection + landmark tensors: one above-threshold anchor yields exactly one
    /// <see cref="DetectedFace"/> whose flattened landmarks have length 10 (5 points) at the expected source pixels.</summary>
    [Fact]
    public unsafe void Decode_ProducesFaceWithTenLandmarkFloats()
    {
        YoloPreprocessor.Transform transform = new YoloPreprocessor(64).ComputeTransform(64, 64); // scale 1, pad 0
        const int anchors = 2;

        Tensor det = new(new TensorShape(1, 4 + NumClasses, anchors), DType.F32);
        Tensor lm = new(new TensorShape(1, NumPoints * KptDims, anchors), DType.F32);
        try
        {
            // Anchor 0 — a real face at box center (20,20) size 10, confidence 0.9. Anchor 1 stays below threshold.
            SetChannel(det, anchors, 0, 0, 20f); // cx
            SetChannel(det, anchors, 1, 0, 20f); // cy
            SetChannel(det, anchors, 2, 0, 10f); // w
            SetChannel(det, anchors, 3, 0, 10f); // h
            SetChannel(det, anchors, 4, 0, 0.9f); // class prob

            // Five landmarks (canvas px == source px here) with score 0.9 each.
            float[,] pts = { { 16, 17 }, { 24, 17 }, { 20, 21 }, { 17, 25 }, { 23, 25 } };
            for (int k = 0; k < NumPoints; k++)
            {
                SetChannel(lm, anchors, k * KptDims + 0, 0, pts[k, 0]);
                SetChannel(lm, anchors, k * KptDims + 1, 0, pts[k, 1]);
                SetChannel(lm, anchors, k * KptDims + 2, 0, 0.9f);
            }

            IReadOnlyList<DetectedFace> faces = FaceDetectionPostProcessor.Decode(
                det, lm, transform, NumClasses, NumPoints, KptDims, confidenceThreshold: 0.25f);

            Assert.Single(faces);
            DetectedFace face = faces[0];
            Assert.Equal(0.9f, face.Confidence, tolerance: 1e-5f);
            Assert.Equal(10, face.Landmarks.ToXyArray().Length);
            Assert.Equal(16f, face.Landmarks.LeftEye.X, tolerance: 1e-3f);
            Assert.Equal(17f, face.Landmarks.LeftEye.Y, tolerance: 1e-3f);
            Assert.Equal(23f, face.Landmarks.MouthRight.X, tolerance: 1e-3f);
            // Box unprojects to xyxy [15,15,25,25].
            Assert.Equal(15f, face.Box.X1, tolerance: 1e-3f);
            Assert.Equal(25f, face.Box.Y2, tolerance: 1e-3f);
        }
        finally
        {
            det.Dispose();
            lm.Dispose();
        }
    }

    /// <summary>NMS collapses two overlapping face boxes to the higher-scoring one.</summary>
    [Fact]
    public unsafe void Decode_NmsDropsOverlappingFaces()
    {
        YoloPreprocessor.Transform transform = new YoloPreprocessor(64).ComputeTransform(64, 64);
        const int anchors = 2;

        Tensor det = new(new TensorShape(1, 4 + NumClasses, anchors), DType.F32);
        Tensor lm = new(new TensorShape(1, NumPoints * KptDims, anchors), DType.F32);
        try
        {
            // Two heavily overlapping boxes (IoU ≈ 0.68): center (20,20) and (21,21), both size 10.
            SetChannel(det, anchors, 0, 0, 20f); SetChannel(det, anchors, 1, 0, 20f);
            SetChannel(det, anchors, 2, 0, 10f); SetChannel(det, anchors, 3, 0, 10f);
            SetChannel(det, anchors, 4, 0, 0.9f);
            SetChannel(det, anchors, 0, 1, 21f); SetChannel(det, anchors, 1, 1, 21f);
            SetChannel(det, anchors, 2, 1, 10f); SetChannel(det, anchors, 3, 1, 10f);
            SetChannel(det, anchors, 4, 1, 0.6f);

            IReadOnlyList<DetectedFace> faces = FaceDetectionPostProcessor.Decode(
                det, lm, transform, NumClasses, NumPoints, KptDims, confidenceThreshold: 0.25f, iouThreshold: 0.45f);

            Assert.Single(faces);
            Assert.Equal(0.9f, faces[0].Confidence, tolerance: 1e-5f);
        }
        finally
        {
            det.Dispose();
            lm.Dispose();
        }
    }

    /// <summary>Real-weight end-to-end parity — env-gated. Set <c>YOLOV8_FACE_PATH</c> to a converted YOLOv8-Face
    /// safetensors to exercise the full pipeline on the GPU lane; unset skips cleanly.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void FaceDetector_RealWeights_RunsOrSkips()
    {
        string? path = Environment.GetEnvironmentVariable("YOLOV8_FACE_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
            return; // Skip: no checkpoint provisioned on this machine.

        using IBackend backend = new CpuBackend();
        using FaceDetector detector = new(backend, YoloV8FaceConfig.YoloV8nFace(), path, inputSize: 640);

        // A 1×1 gray image just proves the forward + decode path runs end-to-end on real weights without throwing;
        // real parity (box/landmark accuracy) belongs to a fixtured image comparison once one is committed.
        int w = 64, h = 64;
        byte[] rgb = new byte[w * h * 3];
        Array.Fill(rgb, (byte)128);
        IReadOnlyList<DetectedFace> faces = detector.DetectFaces(rgb, w, h, confidenceThreshold: 0.25f);
        foreach (DetectedFace f in faces)
            Assert.Equal(10, f.Landmarks.ToXyArray().Length);
    }

    private static unsafe void SetChannel(Tensor t, int anchors, int channel, int anchor, float value)
    {
        float* p = (float*)t.DataPointer;
        p[(long)channel * anchors + anchor] = value;
    }

    private static unsafe Tensor Random4D(int n, int c, int h, int w, int seed)
    {
        Tensor t = new(new TensorShape(n, c, h, w), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++)
            p[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        return t;
    }

    /// <summary>Builds a full synthetic weight dict for <see cref="FaceDetectHead"/> at the given per-scale input
    /// channels, matching the exact keys/shapes <c>LoadWeights</c> reads (cv2 box, cv3 class, cv4 landmark).</summary>
    private static Dictionary<string, Tensor> BuildHeadWeights(int[] inChannels, string prefix)
    {
        int c2 = Math.Max(16, Math.Max(inChannels[0] / 4, RegMax * 4));
        int c3 = Math.Max(inChannels[0], Math.Min(NumClasses, 100));
        int nkFlat = NumPoints * KptDims;
        int c4 = Math.Max(inChannels[0] / 4, nkFlat);
        int boxOut = 4 * RegMax;

        Dictionary<string, Tensor> w = new();
        int seed = 1000;
        for (int s = 0; s < inChannels.Length; s++)
        {
            int cin = inChannels[s];
            // cv2 box branch: two 3×3 stages then a 1×1 projection to 4·reg_max.
            AddConv(w, $"{prefix}.cv2.{s}.0.conv", c2, cin, 3, ref seed);
            AddConv(w, $"{prefix}.cv2.{s}.1.conv", c2, c2, 3, ref seed);
            AddConv(w, $"{prefix}.cv2.{s}.2", boxOut, c2, 1, ref seed);
            // cv3 class branch: two 3×3 stages then a 1×1 projection to nc.
            AddConv(w, $"{prefix}.cv3.{s}.0.conv", c3, cin, 3, ref seed);
            AddConv(w, $"{prefix}.cv3.{s}.1.conv", c3, c3, 3, ref seed);
            AddConv(w, $"{prefix}.cv3.{s}.2", NumClasses, c3, 1, ref seed);
            // cv4 landmark branch: two 3×3 stages then a 1×1 projection to 5·ndim.
            AddConv(w, $"{prefix}.cv4.{s}.0.conv", c4, cin, 3, ref seed);
            AddConv(w, $"{prefix}.cv4.{s}.1.conv", c4, c4, 3, ref seed);
            AddConv(w, $"{prefix}.cv4.{s}.2", nkFlat, c4, 1, ref seed);
        }
        return w;
    }

    private static unsafe void AddConv(Dictionary<string, Tensor> w, string prefix, int outCh, int inCh, int kernel, ref int seed)
    {
        Tensor weight = new(new TensorShape(outCh, inCh, kernel, kernel), DType.F32);
        Tensor bias = new(new TensorShape(outCh), DType.F32);
        FillRandom(weight, seed++);
        FillRandom(bias, seed++);
        w[$"{prefix}.weight"] = weight;
        w[$"{prefix}.bias"] = bias;
    }

    private static unsafe void FillRandom(Tensor t, int seed)
    {
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++)
            p[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
    }
}
