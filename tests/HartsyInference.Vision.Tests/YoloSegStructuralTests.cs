using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Vision.Detection;
using HartsyInference.Vision.Detection.Blocks;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Structural (tiny-config, CPU, synthetic-weight) tests for the YOLOv8-seg building blocks —
/// <see cref="Proto"/>, <see cref="SegmentDetectHead"/>'s mask-coefficient branch, and
/// <see cref="MaskAssembly"/>. These validate the shapes and wiring deterministically without any real
/// checkpoint; real-weight mask-parity vs Ultralytics is the env-gated <c>YoloSegEndToEndTest</c>.</summary>
public sealed unsafe class YoloSegStructuralTests
{
    private static Tensor Filled(TensorShape shape, float value)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = value;
        return t;
    }

    private static Tensor Ramp(TensorShape shape, int seed)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        long n = shape.ElementCount;
        // Small magnitudes keep the synthetic forward numerically tame (no SiLU/exp overflow).
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        return t;
    }

    private static void AddConv(Dictionary<string, Tensor> w, string prefix, int outC, int inC, int k, int seed)
    {
        w[$"{prefix}.weight"] = Ramp(new TensorShape(outC, inC, k, k), seed);
        w[$"{prefix}.bias"] = Filled(new TensorShape(outC), 0f);
    }

    [Fact]
    public void Proto_Forward_DoublesSpatial_AndOutputsNumMasks()
    {
        const int inC = 6, npr = 8, nm = 4, H = 5, W = 5;
        using IBackend backend = new CpuBackend();

        Dictionary<string, Tensor> w = new();
        AddConv(w, "proto.cv1.conv", npr, inC, 3, 1);
        AddConv(w, "proto.cv2.conv", npr, npr, 3, 2);
        AddConv(w, "proto.cv3.conv", nm, npr, 1, 3);
        w["proto.upsample.weight"] = Ramp(new TensorShape(npr, npr, 2, 2), 4);
        w["proto.upsample.bias"] = Filled(new TensorShape(npr), 0f);

        Proto proto = new Proto(inC, npr, nm);
        proto.LoadWeights(w, "proto");
        Assert.Equal(nm, proto.NumMasks);

        using Tensor input = Ramp(new TensorShape(1, inC, H, W), 99);
        using Tensor outp = proto.Forward(backend, input);

        Assert.Equal(4, outp.Shape.Rank);
        Assert.Equal(1, (int)outp.Shape[0]);
        Assert.Equal(nm, (int)outp.Shape[1]);
        Assert.Equal(H * 2, (int)outp.Shape[2]);
        Assert.Equal(W * 2, (int)outp.Shape[3]);
    }

    [Fact]
    public void SegmentDetectHead_Forward_ProducesBoxClassMaskChannels_AndProtos()
    {
        const int nc = 2, nm = 4, regMax = 4, npr = 8;
        int[] inCh = { 6, 12, 24 };
        int[] strides = { 8, 16, 32 };
        // Match the head's internal-channel formulas so the synthetic weight shapes line up.
        int c2 = Math.Max(16, Math.Max(inCh[0] / 4, regMax * 4));
        int c3 = Math.Max(inCh[0], Math.Min(nc, 100));
        int c4 = Math.Max(inCh[0] / 4, nm);

        Dictionary<string, Tensor> w = new();
        int seed = 10;
        for (int s = 0; s < 3; s++)
        {
            AddConv(w, $"head.cv2.{s}.0.conv", c2, inCh[s], 3, seed++);
            AddConv(w, $"head.cv2.{s}.1.conv", c2, c2, 3, seed++);
            AddConv(w, $"head.cv2.{s}.2", 4 * regMax, c2, 1, seed++);
            AddConv(w, $"head.cv3.{s}.0.conv", c3, inCh[s], 3, seed++);
            AddConv(w, $"head.cv3.{s}.1.conv", c3, c3, 3, seed++);
            AddConv(w, $"head.cv3.{s}.2", nc, c3, 1, seed++);
            AddConv(w, $"head.cv4.{s}.0.conv", c4, inCh[s], 3, seed++);
            AddConv(w, $"head.cv4.{s}.1.conv", c4, c4, 3, seed++);
            AddConv(w, $"head.cv4.{s}.2", nm, c4, 1, seed++);
        }
        AddConv(w, "head.proto.cv1.conv", npr, inCh[0], 3, seed++);
        AddConv(w, "head.proto.cv2.conv", npr, npr, 3, seed++);
        AddConv(w, "head.proto.cv3.conv", nm, npr, 1, seed++);
        w["head.proto.upsample.weight"] = Ramp(new TensorShape(npr, npr, 2, 2), seed++);
        w["head.proto.upsample.bias"] = Filled(new TensorShape(npr), 0f);

        SegmentDetectHead head = new SegmentDetectHead(nc, nm, regMax, inCh, strides, npr);
        head.LoadWeights(w, "head");
        Assert.Equal(nm, head.NumMasks);
        Assert.Equal(nc, head.NumClasses);

        using IBackend backend = new CpuBackend();
        // P3 8×8, P4 4×4, P5 2×2 → 64 + 16 + 4 = 84 anchors; protos = 2 × P3 = 16×16.
        using Tensor p3 = Ramp(new TensorShape(1, inCh[0], 8, 8), 200);
        using Tensor p4 = Ramp(new TensorShape(1, inCh[1], 4, 4), 201);
        using Tensor p5 = Ramp(new TensorShape(1, inCh[2], 2, 2), 202);

        (Tensor detections, Tensor protos) = head.Forward(backend, new[] { p3, p4, p5 });
        try
        {
            Assert.Equal(3, detections.Shape.Rank);
            Assert.Equal(1, (int)detections.Shape[0]);
            Assert.Equal(4 + nc + nm, (int)detections.Shape[1]);
            Assert.Equal(84, (int)detections.Shape[2]);

            Assert.Equal(4, protos.Shape.Rank);
            Assert.Equal(nm, (int)protos.Shape[1]);
            Assert.Equal(16, (int)protos.Shape[2]);
            Assert.Equal(16, (int)protos.Shape[3]);

            // Class-probability channels are post-sigmoid → in [0, 1]. Mask-coefficient channels are raw
            // logits (no sigmoid until MaskAssembly), so they must NOT be clamped into [0, 1].
            float* d = (float*)detections.DataPointer;
            int totalAnchors = (int)detections.Shape[2];
            for (int k = 0; k < nc; k++)
                for (int a = 0; a < totalAnchors; a++)
                {
                    float prob = d[(4 + k) * totalAnchors + a];
                    Assert.InRange(prob, 0f, 1f);
                }
        }
        finally
        {
            detections.Dispose();
            protos.Dispose();
        }
    }

    [Fact]
    public void MaskAssembly_CombinesCoefficients_Sigmoid_AndThresholds()
    {
        const int nm = 2, protoH = 8, protoW = 8, src = 8;
        // Layer 0 = strongly positive, layer 1 = strongly negative. coef=[1,0] → +10 everywhere → mask 1.
        using Tensor protos = new Tensor(new TensorShape(1, nm, protoH, protoW), DType.F32);
        float* pp = (float*)protos.DataPointer;
        for (int i = 0; i < protoH * protoW; i++) { pp[i] = 10f; pp[protoH * protoW + i] = -10f; }

        YoloPreprocessor.Transform tr = new(
            SourceWidth: src, SourceHeight: src,
            ResizedWidth: src, ResizedHeight: src,
            Scale: 1f, PadLeft: 0, PadTop: 0,
            PaddedWidth: src, PaddedHeight: src);

        YoloDetection det = new YoloDetection(0f, 0f, src, src, 0.9f, 0);

        byte[] onMask = MaskAssembly.AssembleMask(det, new float[] { 1f, 0f }, protos, tr, threshold: 0.5f);
        Assert.Equal(src * src, onMask.Length);
        int onCount = 0;
        foreach (byte b in onMask) onCount += b;
        Assert.Equal(src * src, onCount);

        // coef=[0,1] selects the strongly-negative layer → sigmoid ≈ 0 → nothing passes threshold.
        byte[] offMask = MaskAssembly.AssembleMask(det, new float[] { 0f, 1f }, protos, tr, threshold: 0.5f);
        int offCount = 0;
        foreach (byte b in offMask) offCount += b;
        Assert.Equal(0, offCount);
    }

    [Fact]
    public void MaskAssembly_CropsToBoundingBox()
    {
        const int nm = 1, protoH = 8, protoW = 8, src = 8;
        using Tensor protos = Filled(new TensorShape(1, nm, protoH, protoW), 10f);

        YoloPreprocessor.Transform tr = new(
            SourceWidth: src, SourceHeight: src,
            ResizedWidth: src, ResizedHeight: src,
            Scale: 1f, PadLeft: 0, PadTop: 0,
            PaddedWidth: src, PaddedHeight: src);

        // Box covers only the top-left 4×4 quadrant; the crop step must zero everything outside it.
        YoloDetection det = new YoloDetection(0f, 0f, 4f, 4f, 0.9f, 0);
        byte[] mask = MaskAssembly.AssembleMask(det, new float[] { 1f }, protos, tr, threshold: 0.5f);

        for (int y = 0; y < src; y++)
            for (int x = 0; x < src; x++)
            {
                byte expected = (x < 4 && y < 4) ? (byte)1 : (byte)0;
                Assert.Equal(expected, mask[y * src + x]);
            }
    }
}
