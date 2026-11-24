using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.DepthPro2;
using HartsyInference.Vision.Dinov3;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests.DepthPro2.Parity;

/// <summary>
/// CPU F32 parity of Depth Pro 2 S and the focal estimator against <c>tests/python-reference/dump_depth_pro2.py</c> (official ml-depth-pro-2 at
/// aa389cbb, timm 1.0.27, fp32). Set <c>HARTSY_DEPTHPRO2_WEIGHTS</c> and <c>HARTSY_DEPTHPRO2_FOCAL_WEIGHTS</c> to the pinned <c>model.safetensors</c> files and
/// <c>HARTSY_DEPTHPRO2_REF</c> to the dump's <c>meta.json</c>; otherwise the tests are reported as skipped. Case <c>a</c> is 256 x 384, <c>b</c> is 250 x 370
/// (replicate-padded to 256 x 384; the estimator resizes it to 224 x 320), <c>c</c> is 256 x 512 (the estimator crops it to 16:9).
/// </summary>
[Trait("Category", "Integration")]
public sealed class DepthPro2ParityTests(ITestOutputHelper output)
{
    private const string WeightsVariable = "HARTSY_DEPTHPRO2_WEIGHTS";
    private const string FocalVariable = "HARTSY_DEPTHPRO2_FOCAL_WEIGHTS";
    private const string ReferenceVariable = "HARTSY_DEPTHPRO2_REF";
    private const string DepthSha256 = "583c4d630779fd231f969c738733624963ad34bd3c29f9c64088257e3436f97c";
    private const string FocalSha256 = "8f567d1c97ff916137288668f6bd306ce2094da2e2b852b0e4539dbb139c8c86";

    // relRMS = rms(ours - reference) / rms(reference). F32 CPU on both sides; the bounds sit above the measured values with room for thread-count drift.
    private const double StageTolerance = 5e-5;
    private const double InverseDepthTolerance = 1e-4;
    private const double FocalTolerance = 1e-4;

    [RealWeightsFact(WeightsVariable, ReferenceVariable)]
    public void RopeTables_MatchTimmGetEmbedWithMaxNormalisation()
    {
        Reference reference = Reference.Open();
        float[] timm = reference.Read("a_rope", out long[] shape);   // [patches, 2·headDim]: sin then cos, as apply_rot_embed_cat splits it
        int patches = (int)shape[0], headDim = (int)shape[1] / 2;
        (Tensor cos, Tensor sin) = Dinov3RopeTables.Build(1, 0, 16, 24, headDim, roundPeriodsToBf16: false, normalizeByLongSide: true);
        using (cos)
        using (sin)
        {
            ReadOnlySpan<float> c = cos.AsReadOnlySpan<float>(), s = sin.AsReadOnlySpan<float>();
            double worst = 0;
            for (int n = 0; n < patches; n++)
                for (int d = 0; d < headDim; d++)
                {
                    worst = Math.Max(worst, Math.Abs(s[n * headDim + d] - timm[n * 2 * headDim + d]));
                    worst = Math.Max(worst, Math.Abs(c[n * headDim + d] - timm[n * 2 * headDim + headDim + d]));
                }

            output.WriteLine($"RoPE 16x24 max abs vs timm get_embed: {worst:E3}");
            Assert.True(worst < 1e-5, $"RoPE table differs from timm by {worst:E3}");
        }
    }

    [RealWeightsFact(WeightsVariable, ReferenceVariable)]
    public void InverseDepth_MatchesReference_256x384() => RunDepth("a");

    [RealWeightsFact(WeightsVariable, ReferenceVariable)]
    public void InverseDepth_MatchesReference_Padded250x370() => RunDepth("b");

    [RealWeightsFact(FocalVariable, ReferenceVariable)]
    public void Focal_MatchesReference_256x384() => RunFocal("a");

    [RealWeightsFact(FocalVariable, ReferenceVariable)]
    public void Focal_MatchesReference_Resized250x370() => RunFocal("b");

    [RealWeightsFact(FocalVariable, ReferenceVariable)]
    public void Focal_MatchesReference_Cropped256x512() => RunFocal("c");

    private void RunDepth(string name)
    {
        Reference reference = Reference.Open();
        string weightsPath = Environment.GetEnvironmentVariable(WeightsVariable)!;
        Assert.Equal(DepthSha256, Sha256(weightsPath), ignoreCase: true);
        using SafeTensorsLoader loader = new();
        loader.Load(weightsPath);
        DepthPro2Network network = new(DepthPro2Preset.Small, loader.GetAllTensors());
        using CpuBackend backend = new();
        List<string> failures = [];

        using Tensor pixels = reference.Tensor($"{name}_pixels");
        using (Tensor? padded = DepthPro2Network.PadToMultiple(backend, pixels))
        {
            Tensor input = padded ?? pixels;
            Compare($"{name}_padded", input, reference, StageTolerance, failures);
            int[] all = Enumerable.Range(0, 12).ToArray();
            Tensor[] blocks = network.Encoder.EncodeIntermediates(backend, input, all, DepthPro2Preset.Small.CrossFrameBlocks);
            try
            {
                for (int i = 0; i < blocks.Length; i++) Compare($"{name}_block{i}", blocks[i], reference, StageTolerance, failures);
            }
            finally
            {
                foreach (Tensor b in blocks) b.Dispose();
            }
        }

        using Tensor inverse = network.PredictInverseDepth(backend, pixels, (stage, t) => Compare($"{name}_{stage}", t, reference, StageTolerance, failures));
        Compare($"{name}_inverse_depth", inverse, reference, InverseDepthTolerance, failures, peak: true);
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    private void RunFocal(string name)
    {
        Reference reference = Reference.Open();
        string weightsPath = Environment.GetEnvironmentVariable(FocalVariable)!;
        Assert.Equal(FocalSha256, Sha256(weightsPath), ignoreCase: true);
        using SafeTensorsLoader loader = new();
        loader.Load(weightsPath);
        DepthPro2FocalEstimator estimator = new(loader.GetAllTensors());
        using CpuBackend backend = new();
        List<string> failures = [];

        using Tensor pixels = reference.Tensor($"{name}_pixels");
        DepthPro2FieldOfView fov = estimator.Estimate(backend, pixels, (stage, t) => Compare($"{name}_{stage}", t, reference, StageTolerance, failures));
        float[] refFov = reference.Read($"{name}_fov_deg", out _);
        float refFocal = reference.Read($"{name}_f_px", out _)[0];
        double focal = DepthPro2FocalEstimator.FocalLengthPx([fov], (int)pixels.Shape[2], (int)pixels.Shape[3]);
        double focalError = Math.Abs(focal - refFocal) / Math.Abs(refFocal);
        output.WriteLine($"{name} fov ({fov.HorizontalDegrees:F5}, {fov.VerticalDegrees:F5}) vs ({refFov[0]:F5}, {refFov[1]:F5}); f_px {focal:F4} vs {refFocal:F4}, "
            + $"rel {focalError:E2}");
        if (Math.Abs(fov.HorizontalDegrees - refFov[0]) / refFov[0] > FocalTolerance || Math.Abs(fov.VerticalDegrees - refFov[1]) / refFov[1] > FocalTolerance)
            failures.Add($"{name} fov differs");
        if (focalError > FocalTolerance) failures.Add($"{name} f_px rel {focalError:E2}");
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    private void Compare(string key, Tensor actual, Reference reference, double tolerance, List<string> failures, bool peak = false)
    {
        float[] expected = reference.Read(key, out long[] shape);
        ReadOnlySpan<float> got = actual.AsReadOnlySpan<float>();
        if (got.Length != expected.Length)
        {
            failures.Add($"{key}: {got.Length} values vs reference {expected.Length} [{string.Join(", ", shape)}]");
            return;
        }

        double diff2 = 0, ref2 = 0, maxDiff = 0, maxRef = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = got[i] - (double)expected[i];
            diff2 += d * d;
            ref2 += (double)expected[i] * expected[i];
            maxDiff = Math.Max(maxDiff, Math.Abs(d));
            maxRef = Math.Max(maxRef, Math.Abs(expected[i]));
        }

        double relRms = Math.Sqrt(diff2 / Math.Max(ref2, 1e-30)), peakRel = maxDiff / Math.Max(maxRef, 1e-30);
        output.WriteLine($"| {key} | {relRms:E2} | {peakRel:E2} |");
        if (relRms > tolerance || (peak && peakRel > tolerance)) failures.Add($"{key}: relRMS {relRms:E2}, peak-relative {peakRel:E2} (bound {tolerance:E0})");
    }

    private static string Sha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream));
    }

    private sealed class Reference
    {
        private readonly string _dir;
        private readonly JsonElement _meta;

        private Reference(string metaPath)
        {
            _dir = Path.GetDirectoryName(metaPath)!;
            _meta = JsonDocument.Parse(File.ReadAllText(metaPath)).RootElement;
        }

        public static Reference Open() => new(Environment.GetEnvironmentVariable(ReferenceVariable)!);

        public float[] Read(string key, out long[] shape)
        {
            shape = _meta.GetProperty(key).EnumerateArray().Select(e => e.GetInt64()).ToArray();
            byte[] bytes = File.ReadAllBytes(Path.Combine(_dir, key + ".bin"));
            float[] values = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }

        public Tensor Tensor(string key)
        {
            float[] values = Read(key, out long[] shape);
            Tensor t = new(new TensorShape(shape), DType.F32);
            values.CopyTo(t.AsSpan<float>());
            return t;
        }
    }
}
