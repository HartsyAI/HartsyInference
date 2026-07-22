using System.Globalization;
using System.Text;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Codec;
using HartsyInference.Vision.Detection;
using HartsyInference.Vision.Face;
using HartsyInference.Vision.FaceDetection;
using Xunit;
using Xunit.Abstractions;

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

    private readonly ITestOutputHelper _output;

    public FaceDetectorForwardTests(ITestOutputHelper output) => _output = output;

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

    /// <summary>Real-weight end-to-end parity — env-gated. Set <c>YOLOV8_FACE_PATH</c> to a folded YOLOv8-Face
    /// safetensors (see <c>tests/python-reference/convert_yolov8_pt_to_safetensors.py</c> on a widerface Pose
    /// checkpoint, e.g. akanametov/yolo-face <c>yolov8n-face.pt</c>). Runs the full detector on a real photo
    /// (<c>TestData/bus.png</c> by default, or <c>FACE_TEST_IMAGE</c>), asserts faces + 5 in-bounds landmarks,
    /// dumps detections to <c>FACE_OUT_JSON</c> for the external oracle diff, and — when <c>ARCFACE_WEIGHTS_PATH</c>
    /// is set — closes the loop to a normalized 512-d ArcFace identity embedding. Unset gates skip cleanly.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void FaceDetector_RealWeights_DetectsFacesLandmarksAndEmbeds()
    {
        string? path = Environment.GetEnvironmentVariable("YOLOV8_FACE_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            _output.WriteLine("SKIPPED: YOLOV8_FACE_PATH unset or missing.");
            return;
        }

        string image = Environment.GetEnvironmentVariable("FACE_TEST_IMAGE")
            ?? Path.Combine(AppContext.BaseDirectory, "TestData", "bus.png");
        if (!File.Exists(image))
        {
            _output.WriteLine($"SKIPPED: test image not found: {image}");
            return;
        }

        (byte[] rgb, int w, int h) = PngDecoder.DecodeFromFile(image);

        using IBackend backend = new CpuBackend();
        using FaceDetector detector = new(backend, YoloV8FaceConfig.YoloV8nFace(), path, inputSize: 640);
        IReadOnlyList<DetectedFace> faces = detector.DetectFaces(rgb, w, h, confidenceThreshold: 0.25f, iouThreshold: 0.45f);

        _output.WriteLine($"Detected {faces.Count} faces on {Path.GetFileName(image)} ({w}x{h})");
        Assert.NotEmpty(faces);
        foreach (DetectedFace f in faces)
        {
            Assert.Equal(NumPoints * 2, f.Landmarks.ToXyArray().Length);
            Assert.InRange(f.Box.X1, 0f, w);
            Assert.InRange(f.Box.Y1, 0f, h);
            Assert.InRange(f.Box.X2, 0f, w);
            Assert.InRange(f.Box.Y2, 0f, h);
            Assert.True(f.Box.X2 > f.Box.X1 && f.Box.Y2 > f.Box.Y1, $"Degenerate box: {f.Box}");
            for (int k = 0; k < Landmark5.Count; k++)
            {
                Assert.InRange(f.Landmarks[k].X, 0f, w);
                Assert.InRange(f.Landmarks[k].Y, 0f, h);
            }
            _output.WriteLine($"  conf={f.Confidence:F3} box=({f.Box.X1:F1},{f.Box.Y1:F1},{f.Box.X2:F1},{f.Box.Y2:F1}) " +
                $"eyeL=({f.Landmarks.LeftEye.X:F1},{f.Landmarks.LeftEye.Y:F1}) nose=({f.Landmarks.Nose.X:F1},{f.Landmarks.Nose.Y:F1})");
        }

        string? outJson = Environment.GetEnvironmentVariable("FACE_OUT_JSON");
        if (!string.IsNullOrEmpty(outJson))
        {
            WriteDetectionsJson(outJson, w, h, faces);
            _output.WriteLine($"Wrote C# detections to {outJson}");
        }

        // ArcFace closes detector → 5 landmarks → template align → 512-d identity embedding.
        string? arc = Environment.GetEnvironmentVariable("ARCFACE_WEIGHTS_PATH")
            ?? Environment.GetEnvironmentVariable("ARCFACE_WEIGHTS");
        if (!string.IsNullOrEmpty(arc) && File.Exists(arc))
        {
            using SafeTensorsLoader arcLoader = new();
            arcLoader.Load(arc);
            ArcFaceModel arcFace = new();
            arcFace.LoadWeights(arcLoader.GetAllTensors());
            float[] emb = FaceDetector.EmbedFace(backend, arcFace, rgb, w, h, faces[0]);
            Assert.Equal(ArcFaceModel.EmbeddingDim, emb.Length);
            double norm = 0;
            foreach (float v in emb)
            {
                Assert.True(float.IsFinite(v), "ArcFace embedding has a non-finite value.");
                norm += (double)v * v;
            }
            norm = Math.Sqrt(norm);
            Assert.Equal(1.0, norm, 2); // EmbedNormalized L2-normalizes.
            _output.WriteLine($"ArcFace embedding: dim={emb.Length} L2-norm={norm:F5} first4=[{emb[0]:F4},{emb[1]:F4},{emb[2]:F4},{emb[3]:F4}]");
        }
        else
        {
            _output.WriteLine("ArcFace embedding step skipped (ARCFACE_WEIGHTS_PATH unset).");
        }
    }

    /// <summary>Serializes detections to a small JSON the Python oracle-diff reads (box xyxy + 5 landmarks in source px).</summary>
    private static void WriteDetectionsJson(string outPath, int w, int h, IReadOnlyList<DetectedFace> faces)
    {
        CultureInfo ci = CultureInfo.InvariantCulture;
        StringBuilder sb = new();
        sb.Append("{\"src_w\":").Append(w).Append(",\"src_h\":").Append(h).Append(",\"detections\":[");
        for (int i = 0; i < faces.Count; i++)
        {
            DetectedFace f = faces[i];
            if (i > 0) sb.Append(',');
            sb.Append("{\"box\":[")
              .Append(f.Box.X1.ToString("R", ci)).Append(',').Append(f.Box.Y1.ToString("R", ci)).Append(',')
              .Append(f.Box.X2.ToString("R", ci)).Append(',').Append(f.Box.Y2.ToString("R", ci)).Append(']')
              .Append(",\"conf\":").Append(f.Confidence.ToString("R", ci))
              .Append(",\"landmarks\":[");
            for (int k = 0; k < Landmark5.Count; k++)
            {
                if (k > 0) sb.Append(',');
                FaceLandmark p = f.Landmarks[k];
                sb.Append('[').Append(p.X.ToString("R", ci)).Append(',').Append(p.Y.ToString("R", ci))
                  .Append(',').Append(p.Score.ToString("R", ci)).Append(']');
            }
            sb.Append("]}");
        }
        sb.Append("]}");
        File.WriteAllText(outPath, sb.ToString());
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
