using System;
using System.IO;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Vision.Detection.GroundingDino;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests;

/// <summary>Real-weight numerical parity for Grounding DINO (<c>IDEA-Research/grounding-dino-tiny</c>) against a
/// Hugging Face <c>GroundingDinoForObjectDetection</c> oracle. Gated on <c>GROUNDING_DINO_PATH</c> (the
/// <c>model.safetensors</c>) and <c>GROUNDING_DINO_REF</c> (raw-binary boundary dumps from <c>oracle.py</c>).
/// Validates each ported stage against the corresponding dumped intermediate.</summary>
public sealed class GroundingDinoRealWeightParityTests
{
    private readonly ITestOutputHelper _out;

    public GroundingDinoRealWeightParityTests(ITestOutputHelper output) => _out = output;

    [Trait("Category", "Integration")]
    [Fact]
    public void TextTower_MatchesReference()
    {
        if (!TryPaths(out string ckpt, out string refDir))
            return;

        using IBackend backend = new CpuBackend();
        using SafeTensorsLoader loader = new();
        loader.Load(ckpt);
        System.Collections.Generic.Dictionary<string, Tensor> w = loader.GetAllTensors();

        int[] ids = LoadI64(Path.Combine(refDir, "input_input_ids.i64"));
        GroundingDinoConfig cfg = GroundingDinoConfig.Tiny;
        using GroundingDinoTextTower tower = new(cfg);
        tower.LoadWeights(w, "model");

        using Tensor got = tower.Encode(backend, ids);
        float[] reference = LoadF32(Path.Combine(refDir, "text_features.bin"));
        (double corr, double maxAbs) = Compare(got, reference);
        _out.WriteLine($"text_features: corr={corr:F6} maxAbs={maxAbs:E3} (n={reference.Length}, T={ids.Length})");
        Assert.True(corr > 0.999, $"text-tower correlation too low: {corr:F6}");
        Assert.True(maxAbs < 5e-3, $"text-tower max-abs diff too high: {maxAbs:E3}");
    }

    [Trait("Category", "Integration")]
    [Fact]
    public void SwinBackbone_MatchesReference()
    {
        if (!TryPaths(out string ckpt, out string refDir))
            return;

        using IBackend backend = new CpuBackend();
        using SafeTensorsLoader loader = new();
        loader.Load(ckpt);
        System.Collections.Generic.Dictionary<string, Tensor> w = loader.GetAllTensors();

        GroundingDinoConfig cfg = GroundingDinoConfig.Tiny;
        using SwinBackbone swin = new(cfg);
        swin.LoadWeights(w);

        using Tensor pixels = LoadF32Tensor(Path.Combine(refDir, "input_pixel_values.bin"), new TensorShape(1, 3, 800, 1066));
        Tensor[] feats = swin.Forward(backend, pixels);
        try
        {
            for (int i = 0; i < feats.Length; i++)
            {
                float[] reference = LoadF32(Path.Combine(refDir, $"backbone_feat_{i}.bin"));
                (double corr, double maxAbs) = Compare(feats[i], reference);
                _out.WriteLine($"backbone_feat_{i} {feats[i].Shape}: corr={corr:F6} maxAbs={maxAbs:E3} (n={reference.Length})");
                Assert.True(corr > 0.99, $"backbone_feat_{i} correlation too low: {corr:F6}");
            }
        }
        finally
        {
            foreach (Tensor t in feats) t.Dispose();
        }
    }

    [Trait("Category", "Integration")]
    [Fact]
    public void Neck_MatchesReference()
    {
        if (!TryPaths(out string ckpt, out string refDir))
            return;

        using IBackend backend = new CpuBackend();
        using SafeTensorsLoader loader = new();
        loader.Load(ckpt);
        System.Collections.Generic.Dictionary<string, Tensor> w = loader.GetAllTensors();

        GroundingDinoConfig cfg = GroundingDinoConfig.Tiny;
        using GroundingDinoNeck neck = new(cfg);
        neck.LoadWeights(w);

        Tensor[] feats =
        [
            LoadF32Tensor(Path.Combine(refDir, "backbone_feat_0.bin"), new TensorShape(1, 192, 100, 134)),
            LoadF32Tensor(Path.Combine(refDir, "backbone_feat_1.bin"), new TensorShape(1, 384, 50, 67)),
            LoadF32Tensor(Path.Combine(refDir, "backbone_feat_2.bin"), new TensorShape(1, 768, 25, 34)),
        ];
        GroundingDinoNeck.Result r = neck.Forward(backend, feats);
        foreach (Tensor t in feats) t.Dispose();

        float[] refSource = LoadF32(Path.Combine(refDir, "enc_in_vision.bin"));
        float[] refPos = LoadF32(Path.Combine(refDir, "enc_in_vpos.bin"));
        (double cS, double mS) = Compare(r.SourceFlatten, refSource);
        (double cP, double mP) = Compare(r.PositionFlatten, refPos);
        _out.WriteLine($"source_flatten: corr={cS:F6} maxAbs={mS:E3} (tokens={r.TotalTokens})");
        _out.WriteLine($"lvl_pos_embed:  corr={cP:F6} maxAbs={mP:E3}");
        r.SourceFlatten.Dispose();
        r.PositionFlatten.Dispose();
        Assert.True(cS > 0.999, $"source_flatten corr too low: {cS:F6}");
        Assert.True(cP > 0.999, $"lvl_pos_embed corr too low: {cP:F6}");
    }

    [Trait("Category", "Integration")]
    [Fact]
    public void Encoder_MatchesReference()
    {
        if (!TryPaths(out string ckpt, out string refDir))
            return;

        using IBackend backend = new CpuBackend();
        using SafeTensorsLoader loader = new();
        loader.Load(ckpt);
        System.Collections.Generic.Dictionary<string, Tensor> w = loader.GetAllTensors();

        GroundingDinoConfig cfg = GroundingDinoConfig.Tiny;
        int[] ids = LoadI64(Path.Combine(refDir, "input_input_ids.i64"));
        (bool[] attend, int[] positionIds) = GroundingDinoTextPrompt.Build(ids);
        int[][] shapes = [[100, 134], [50, 67], [25, 34], [13, 17]];
        int[] levelStart = [0, 13400, 16750, 17600];
        int nimg = 17821, t = ids.Length;

        using Tensor vision = LoadF32Tensor(Path.Combine(refDir, "enc_in_vision.bin"), new TensorShape(1, nimg, cfg.DModel));
        using Tensor vpos = LoadF32Tensor(Path.Combine(refDir, "enc_in_vpos.bin"), new TensorShape(1, nimg, cfg.DModel));
        using Tensor text = LoadF32Tensor(Path.Combine(refDir, "text_features.bin"), new TensorShape(1, t, cfg.DModel));

        using GroundingDinoEncoder enc = new(cfg);
        enc.LoadWeights(w);
        (Tensor ev, Tensor et) = enc.Forward(backend, vision, vpos, text, attend, positionIds, shapes, levelStart);

        float[] refV = LoadF32(Path.Combine(refDir, "encoder_vision.bin"));
        float[] refT = LoadF32(Path.Combine(refDir, "encoder_text.bin"));
        (double cV, double mV) = Compare(ev, refV);
        (double cT, double mT) = Compare(et, refT);
        _out.WriteLine($"encoder_vision: corr={cV:F6} maxAbs={mV:E3} (n={refV.Length})");
        _out.WriteLine($"encoder_text:   corr={cT:F6} maxAbs={mT:E3} (n={refT.Length})");
        ev.Dispose(); et.Dispose();
        Assert.True(cV > 0.99, $"encoder_vision corr too low: {cV:F6}");
        Assert.True(cT > 0.99, $"encoder_text corr too low: {cT:F6}");
    }

    [Trait("Category", "Integration")]
    [Fact]
    public void DetectorFromEncoderDumps_MatchesReference()
    {
        if (!TryPaths(out string ckpt, out string refDir))
            return;

        using IBackend backend = new CpuBackend();
        using SafeTensorsLoader loader = new();
        loader.Load(ckpt);
        System.Collections.Generic.Dictionary<string, Tensor> w = loader.GetAllTensors();

        GroundingDinoConfig cfg = GroundingDinoConfig.Tiny;
        int[] ids = LoadI64(Path.Combine(refDir, "input_input_ids.i64"));
        int t = ids.Length;
        int[][] shapes = [[100, 134], [50, 67], [25, 34], [13, 17]];
        int[] levelStart = [0, 13400, 16750, 17600];

        using Tensor encVision = LoadF32Tensor(Path.Combine(refDir, "encoder_vision.bin"), new TensorShape(1, 17821, cfg.DModel));
        using Tensor encText = LoadF32Tensor(Path.Combine(refDir, "encoder_text.bin"), new TensorShape(1, t, cfg.DModel));

        using GroundingDinoDetector detector = new(cfg);
        detector.LoadWeights(w);
        GroundingDinoDetector.Output outp = detector.Forward(backend, encVision, encText, t, shapes, levelStart);

        // vocab for phrase decoding
        string vocabPath = Path.Combine(Path.GetDirectoryName(ckpt)!, "vocab.txt");
        string[] vocab = File.Exists(vocabPath) ? GroundingDinoPipeline.LoadVocab(vocabPath) : Array.Empty<string>();
        List<GroundingDinoDetection> dets = GroundingDinoPipeline.PostProcess(outp.Logits, outp.PredBoxes, ids, vocab, 480, 640);
        outp.Logits.Dispose();
        outp.PredBoxes.Dispose();

        _out.WriteLine($"C# detections ({dets.Count}):");
        foreach (GroundingDinoDetection dd in dets)
            _out.WriteLine($"  [{dd.X0:F1},{dd.Y0:F1},{dd.X1:F1},{dd.Y1:F1}] score={dd.Score:F4} '{dd.Label}'");

        // Oracle detections (from manifest.json)
        (float[] box, float score, string label)[] oracle =
        [
            ([344.541f, 23.179f, 637.324f, 374.527f], 0.4878f, "a cat"),
            ([12.228f, 52.015f, 316.895f, 472.613f], 0.4505f, "a cat"),
            ([38.775f, 70.055f, 176.653f, 118.041f], 0.4651f, "a remote control"),
        ];

        foreach ((float[] ob, float os, string ol) in oracle)
        {
            double bestIou = 0; float bestScore = 0; string bestLabel = "";
            foreach (GroundingDinoDetection dd in dets)
            {
                double iou = Iou(ob, [dd.X0, dd.Y0, dd.X1, dd.Y1]);
                if (iou > bestIou) { bestIou = iou; bestScore = dd.Score; bestLabel = dd.Label; }
            }
            _out.WriteLine($"oracle '{ol}' s={os:F3} -> bestIoU={bestIou:F4} score={bestScore:F4} label='{bestLabel}'");
            Assert.True(bestIou >= 0.9, $"top-box IoU too low for '{ol}': {bestIou:F4}");
            Assert.Equal(ol, bestLabel);
            Assert.True(Math.Abs(bestScore - os) < 0.05, $"score off for '{ol}': {bestScore:F4} vs {os:F4}");
        }
    }

    [Trait("Category", "Integration")]
    [Fact]
    public void EndToEnd_MatchesReference()
    {
        if (!TryPaths(out string ckpt, out string refDir))
            return;

        using IBackend backend = new CpuBackend();
        using SafeTensorsLoader loader = new();
        loader.Load(ckpt);
        System.Collections.Generic.Dictionary<string, Tensor> w = loader.GetAllTensors();

        GroundingDinoConfig cfg = GroundingDinoConfig.Tiny;
        int[] ids = LoadI64(Path.Combine(refDir, "input_input_ids.i64"));
        using Tensor pixels = LoadF32Tensor(Path.Combine(refDir, "input_pixel_values.bin"), new TensorShape(1, 3, 800, 1066));

        using GroundingDinoModel model = new(cfg);
        model.LoadWeights(w);
        GroundingDinoDetector.Output outp = model.Forward(backend, pixels, ids);

        string vocabPath = Path.Combine(Path.GetDirectoryName(ckpt)!, "vocab.txt");
        string[] vocab = File.Exists(vocabPath) ? GroundingDinoPipeline.LoadVocab(vocabPath) : Array.Empty<string>();
        List<GroundingDinoDetection> dets = GroundingDinoPipeline.PostProcess(outp.Logits, outp.PredBoxes, ids, vocab, 480, 640);
        outp.Logits.Dispose();
        outp.PredBoxes.Dispose();

        _out.WriteLine($"C# end-to-end detections ({dets.Count}):");
        foreach (GroundingDinoDetection dd in dets)
            _out.WriteLine($"  [{dd.X0:F1},{dd.Y0:F1},{dd.X1:F1},{dd.Y1:F1}] score={dd.Score:F4} '{dd.Label}'");

        (float[] box, float score, string label)[] oracle =
        [
            ([344.541f, 23.179f, 637.324f, 374.527f], 0.4878f, "a cat"),
            ([12.228f, 52.015f, 316.895f, 472.613f], 0.4505f, "a cat"),
            ([38.775f, 70.055f, 176.653f, 118.041f], 0.4651f, "a remote control"),
        ];
        foreach ((float[] ob, float os, string ol) in oracle)
        {
            double bestIou = 0; float bestScore = 0; string bestLabel = "";
            foreach (GroundingDinoDetection dd in dets)
            {
                double iou = Iou(ob, [dd.X0, dd.Y0, dd.X1, dd.Y1]);
                if (iou > bestIou) { bestIou = iou; bestScore = dd.Score; bestLabel = dd.Label; }
            }
            _out.WriteLine($"oracle '{ol}' s={os:F3} -> bestIoU={bestIou:F4} score={bestScore:F4} label='{bestLabel}'");
            Assert.True(bestIou >= 0.9, $"end-to-end top-box IoU too low for '{ol}': {bestIou:F4}");
            Assert.Equal(ol, bestLabel);
            Assert.True(Math.Abs(bestScore - os) < 0.06, $"end-to-end score off for '{ol}': {bestScore:F4} vs {os:F4}");
        }
    }

    private static double Iou(float[] a, float[] b)
    {
        float ix0 = Math.Max(a[0], b[0]), iy0 = Math.Max(a[1], b[1]);
        float ix1 = Math.Min(a[2], b[2]), iy1 = Math.Min(a[3], b[3]);
        float iw = Math.Max(0, ix1 - ix0), ih = Math.Max(0, iy1 - iy0);
        float inter = iw * ih;
        float areaA = (a[2] - a[0]) * (a[3] - a[1]);
        float areaB = (b[2] - b[0]) * (b[3] - b[1]);
        return inter / (areaA + areaB - inter + 1e-9);
    }

    private static unsafe Tensor LoadF32Tensor(string path, TensorShape shape)
    {
        float[] data = LoadF32(path);
        Tensor t = new(shape, DType.F32);
        if (data.Length != t.ElementCount)
            throw new InvalidOperationException($"size mismatch loading {path}: {data.Length} vs {t.ElementCount}");
        fixed (float* src = data)
            Buffer.MemoryCopy(src, (void*)t.DataPointer, (long)data.Length * 4, (long)data.Length * 4);
        return t;
    }

    private static bool TryPaths(out string ckpt, out string refDir)
    {
        ckpt = Environment.GetEnvironmentVariable("GROUNDING_DINO_PATH") ?? "";
        refDir = Environment.GetEnvironmentVariable("GROUNDING_DINO_REF") ?? "";
        return !string.IsNullOrEmpty(ckpt) && File.Exists(ckpt)
            && !string.IsNullOrEmpty(refDir) && Directory.Exists(refDir);
    }

    private static int[] LoadI64(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        int[] outv = new int[bytes.Length / 8];
        for (int i = 0; i < outv.Length; i++)
            outv[i] = (int)BitConverter.ToInt64(bytes, i * 8);
        return outv;
    }

    private static float[] LoadF32(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        float[] outv = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, outv, 0, bytes.Length);
        return outv;
    }

    private static unsafe (double corr, double maxAbs) Compare(Tensor got, float[] reference)
    {
        long n = got.ElementCount;
        if (n != reference.Length)
            throw new InvalidOperationException($"element count mismatch: got {n} vs ref {reference.Length}");
        float* g = (float*)got.DataPointer;
        double sumG = 0, sumR = 0, sumGG = 0, sumRR = 0, sumGR = 0, maxAbs = 0;
        for (long i = 0; i < n; i++)
        {
            double a = g[i], b = reference[i];
            sumG += a; sumR += b; sumGG += a * a; sumRR += b * b; sumGR += a * b;
            double d = Math.Abs(a - b);
            if (d > maxAbs) maxAbs = d;
        }
        double covar = sumGR - sumG * sumR / n;
        double varG = sumGG - sumG * sumG / n;
        double varR = sumRR - sumR * sumR / n;
        double corr = covar / Math.Sqrt(varG * varR + 1e-12);
        return (corr, maxAbs);
    }
}
