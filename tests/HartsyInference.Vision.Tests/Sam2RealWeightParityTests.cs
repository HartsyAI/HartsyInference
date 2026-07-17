using System;
using System.Globalization;
using System.IO;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Vision.Segmentation;
using HartsyInference.Vision.Segmentation.Sam2;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests;

/// <summary>Real-weight numerical parity for SAM 2 (Hiera-tiny) against a Hugging Face <c>Sam2Model</c> oracle.
/// Gated on <c>SAM2_CHECKPOINT_PATH</c> (a converted single-file safetensors of <c>sam2_hiera_tiny.pt</c>) and
/// <c>SAM2_REF_DIR</c> (raw-binary reference dumps produced by <c>sam2_oracle.py</c>). Compares the Hiera image
/// embedding (corr / maxAbs) and the predicted best-mask IoU vs the reference mask on a fixed image + point.</summary>
public sealed class Sam2RealWeightParityTests
{
    private const int Size = 1024;

    private readonly ITestOutputHelper _out;

    public Sam2RealWeightParityTests(ITestOutputHelper output) => _out = output;

    [Trait("Category", "Integration")]
    [Fact]
    public void Sam2Tiny_ImageEmbeddingAndMask_MatchReference()
    {
        string? ckpt = Environment.GetEnvironmentVariable("SAM2_CHECKPOINT_PATH");
        string? refDir = Environment.GetEnvironmentVariable("SAM2_REF_DIR");
        if (string.IsNullOrEmpty(ckpt) || !File.Exists(ckpt) || string.IsNullOrEmpty(refDir) || !Directory.Exists(refDir))
        {
            return; // clean skip: no checkpoint / reference
        }

        using IBackend backend = new CpuBackend();
        Sam2Config cfg = Sam2Config.HieraTiny;

        (Sam2Converter.Groups groups, HartsyInference.ModelHandler.SafeTensors.SafeTensorsLoader loader) =
            Sam2Converter.LoadAndConvert(ckpt);
        using (loader)
        {
            HieraImageEncoder encoder = new HieraImageEncoder(cfg);
            encoder.LoadWeights(groups.ImageEncoder, string.Empty);
            SamPromptEncoder promptEncoder = SamPipeline.BuildPromptEncoder(groups.PromptEncoder);
            SamMaskDecoder decoder = new SamMaskDecoder(cfg);
            decoder.LoadWeights(groups.MaskDecoder, string.Empty);
            using SamPipeline pipeline = new SamPipeline(backend, cfg, encoder, promptEncoder, decoder);

            Tensor image = LoadFloatTensor(Path.Combine(refDir, "pixel_values.bin"), new TensorShape(1, 3, Size, Size));

            // ── Component parity: Hiera image embedding vs reference vision_features [1,256,64,64].
            SamImageFeatures feats = encoder.Forward(backend, image);
            float[] refVf = LoadFloats(Path.Combine(refDir, "vision_features.bin"));
            (double corr, double maxAbs) = Compare(feats.VisionFeatures, refVf);
            _out.WriteLine($"vision_features: corr={corr:F6} maxAbs={maxAbs:E3} (n={refVf.Length})");
            Assert.True(corr > 0.99, $"image-embedding correlation too low: {corr:F6}");

            // ── End-to-end: best-mask IoU vs reference mask on the fixed foreground point.
            SamPrompt prompt = new SamPrompt { Points = [new SamPointPrompt(620, 470, Foreground: true)] };
            SamMaskResult[] results = pipeline.Segment(prompt, image, Size, Size);
            Assert.NotEmpty(results);

            byte[] refMask = File.ReadAllBytes(Path.Combine(refDir, "best_mask.bin"));
            Assert.Equal(Size * Size, refMask.Length);
            // results are best-first by IoU score; result[0] is the model's chosen mask.
            double iou = MaskIoU(results[0].Mask, refMask);
            _out.WriteLine($"best-mask: score={results[0].IoUScore:F4} IoU-vs-ref={iou:F4} " +
                           $"(scores={string.Join(",", Array.ConvertAll(results, r => r.IoUScore.ToString("F3", CultureInfo.InvariantCulture)))})");
            Assert.True(iou >= 0.9, $"best-mask IoU vs reference too low: {iou:F4}");
        }
    }

    private static unsafe Tensor LoadFloatTensor(string path, TensorShape shape)
    {
        float[] data = LoadFloats(path);
        if (data.Length != shape.ElementCount)
            throw new InvalidOperationException($"{path}: {data.Length} floats != shape {shape} ({shape.ElementCount}).");
        Tensor t = new Tensor(shape, DType.F32);
        Span<float> dst = t.AsSpan<float>();
        data.AsSpan().CopyTo(dst);
        return t;
    }

    private static float[] LoadFloats(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        float[] f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, f.Length * 4);
        return f;
    }

    private static unsafe (double corr, double maxAbs) Compare(Tensor got, float[] reference)
    {
        Tensor f = got.DType == DType.F32 ? got : got.CastTo(DType.F32);
        ReadOnlySpan<float> a = f.AsReadOnlySpan<float>();
        if (a.Length != reference.Length)
            throw new InvalidOperationException($"length mismatch {a.Length} vs {reference.Length}.");
        double sa = 0, sb = 0;
        for (int i = 0; i < a.Length; i++) { sa += a[i]; sb += reference[i]; }
        double ma = sa / a.Length, mb = sb / a.Length;
        double cov = 0, va = 0, vb = 0, maxAbs = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double da = a[i] - ma, db = reference[i] - mb;
            cov += da * db; va += da * da; vb += db * db;
            double ad = Math.Abs(a[i] - reference[i]);
            if (ad > maxAbs) maxAbs = ad;
        }
        double corr = cov / (Math.Sqrt(va * vb) + 1e-12);
        return (corr, maxAbs);
    }

    private static double MaskIoU(byte[] a, byte[] b)
    {
        long inter = 0, union = 0;
        for (int i = 0; i < a.Length; i++)
        {
            bool x = a[i] != 0, y = b[i] != 0;
            if (x || y) union++;
            if (x && y) inter++;
        }
        return union == 0 ? 1.0 : (double)inter / union;
    }
}
