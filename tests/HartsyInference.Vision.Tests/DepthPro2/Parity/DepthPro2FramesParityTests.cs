using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.DepthPro2;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests.DepthPro2.Parity;

/// <summary>
/// Depth Pro 2 S cross-frame attention and video on CPU F32 against <c>tests/python-reference/dump_depth_pro2_frames.py</c> (official
/// ml-depth-pro-2 at aa389cbb, timm 1.0.27, fp32): a T = 3 joint forward block by block and through <c>infer</c>, buffered <c>infer_video</c>
/// (window 4, overlap 2) on 6, 7 and 3 frames, and unbuffered <c>infer_video</c> with a 16-frame cache (partial caches only) and a 3-frame cache
/// (the ring wraps) on 6 frames. The frames are 100 x 150 (replicate-padded to 128 x 160). Needs <c>HARTSY_DEPTHPRO2_WEIGHTS</c>,
/// <c>HARTSY_DEPTHPRO2_FOCAL_WEIGHTS</c> and <c>HARTSY_DEPTHPRO2_FRAMES_REF</c> (the dump's <c>meta.json</c>); otherwise reported as skipped.
/// </summary>
[Trait("Category", "Integration")]
public sealed class DepthPro2FramesParityTests(ITestOutputHelper output)
{
    private const string WeightsVariable = "HARTSY_DEPTHPRO2_WEIGHTS";
    private const string FocalVariable = "HARTSY_DEPTHPRO2_FOCAL_WEIGHTS";
    private const string ReferenceVariable = "HARTSY_DEPTHPRO2_FRAMES_REF";
    private const double StageTolerance = 5e-5;
    private const double InverseDepthTolerance = 1e-4;
    private const double FocalTolerance = 1e-4;

    [RealWeightsFact(WeightsVariable, ReferenceVariable)]
    public void JointForward_EveryBlockMatches_T3()
    {
        Reference reference = Reference.Open();
        (byte[][] frames, int height, int width) = reference.Frames(3);
        using SafeTensorsLoader loader = Load(WeightsVariable);
        DepthPro2Network network = new(DepthPro2Preset.Small, loader.GetAllTensors());
        using CpuBackend backend = new();
        using Tensor pixels = DepthPro2Preprocessor.NormalizeFrames(frames, width, height);
        using Tensor? padded = DepthPro2Network.PadToMultiple(backend, pixels);

        int[] every = [.. Enumerable.Range(0, 12)];
        Tensor[] blocks = network.Encoder.EncodeFrameIntermediates(backend, padded ?? pixels, every, DepthPro2Preset.Small.CrossFrameBlocks);
        List<string> failures = [];
        try
        {
            for (int i = 0; i < blocks.Length; i++)
            {
                double rel = RelRms(blocks[i].AsReadOnlySpan<float>(), reference.Read($"joint_block{i}", out long[] shape));
                Assert.Equal(shape, new[] { blocks[i].Shape[0], blocks[i].Shape[1], blocks[i].Shape[2] });
                output.WriteLine($"block {i,2}{(DepthPro2Preset.Small.CrossFrameBlocks.Contains(i) ? " (cross-frame)" : "")} relRMS {rel:E2}");
                if (rel > StageTolerance) failures.Add($"block {i}: {rel:E2}");
            }
        }
        finally
        {
            foreach (Tensor t in blocks) t.Dispose();
        }

        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    [RealWeightsFact(WeightsVariable, FocalVariable, ReferenceVariable)]
    public void JointForward_MatchesInferOnAClip_T3()
    {
        Reference reference = Reference.Open();
        (byte[][] frames, int height, int width) = reference.Frames(3);
        using SafeTensorsLoader depthLoader = Load(WeightsVariable);
        using SafeTensorsLoader focalLoader = Load(FocalVariable);
        DepthPro2Model model = new(new DepthPro2Network(DepthPro2Preset.Small, depthLoader.GetAllTensors()));
        using CpuBackend backend = new();

        DepthPro2Prediction prediction = model.Predict(backend, frames, width, height, null, new DepthPro2FocalEstimator(focalLoader.GetAllTensors()));

        CheckFocal("joint", prediction.FocalLengthPx, reference);
        CheckMaps("joint", prediction.Maps, reference);
    }

    [RealWeightsFact(WeightsVariable, FocalVariable, ReferenceVariable)]
    public void Buffered_MatchesInferVideo_SixFrames() => RunVideo("buffered6", 6, 4, 2, DepthPro2VideoMode.Buffered);

    [RealWeightsFact(WeightsVariable, FocalVariable, ReferenceVariable)]
    public void Buffered_MatchesInferVideo_SevenFramesWithAPulledBackLastClip() => RunVideo("buffered7", 7, 4, 2, DepthPro2VideoMode.Buffered);

    [RealWeightsFact(WeightsVariable, FocalVariable, ReferenceVariable)]
    public void Buffered_MatchesInferVideo_ShorterThanOneWindow() => RunVideo("buffered3", 3, 4, 2, DepthPro2VideoMode.Buffered);

    [RealWeightsFact(WeightsVariable, FocalVariable, ReferenceVariable)]
    public void Buffered_MatchesInferVideo_NoOverlapShortTailClip() => RunVideo("buffered6k0", 6, 4, 0, DepthPro2VideoMode.Buffered);

    [RealWeightsFact(WeightsVariable, FocalVariable, ReferenceVariable)]
    public void Unbuffered_MatchesInferVideo_PartialCaches() => RunVideo("unbuffered16", 6, 16, 0, DepthPro2VideoMode.Unbuffered);

    [RealWeightsFact(WeightsVariable, FocalVariable, ReferenceVariable)]
    public void Unbuffered_MatchesInferVideo_WrappingRing() => RunVideo("unbuffered3", 6, 3, 0, DepthPro2VideoMode.Unbuffered);

    private void RunVideo(string name, int count, int window, int overlap, DepthPro2VideoMode mode)
    {
        Reference reference = Reference.Open();
        (byte[][] frames, int height, int width) = reference.Frames(count);
        using SafeTensorsLoader depthLoader = Load(WeightsVariable);
        using SafeTensorsLoader focalLoader = Load(FocalVariable);
        DepthPro2Network network = new(DepthPro2Preset.Small, depthLoader.GetAllTensors());
        using CpuBackend backend = new();
        double focal = DepthPro2Model.EstimateFocal(backend, frames, window, width, height, new DepthPro2FocalEstimator(focalLoader.GetAllTensors()));
        CheckFocal(name, focal, reference);

        List<DepthPro2Map> maps = [];
        List<string> emits = [];
        using (DepthPro2VideoStream stream = new(network, backend, width, height, focal, window, overlap, mode))
        {
            foreach (byte[] frame in frames)
            {
                IReadOnlyList<DepthPro2Map> done = stream.Step(frame);
                emits.Add(done.Count.ToString(System.Globalization.CultureInfo.InvariantCulture));
                maps.AddRange(done);
            }

            IReadOnlyList<DepthPro2Map> rest = stream.Flush();
            emits.Add("flush " + rest.Count.ToString(System.Globalization.CultureInfo.InvariantCulture));
            maps.AddRange(rest);
            Assert.Throws<InvalidOperationException>(() => stream.Step(frames[0]));
            Assert.Throws<InvalidOperationException>(() => stream.Flush());
        }

        output.WriteLine($"{name}: maps per step {string.Join(", ", emits)}");
        CheckMaps(name, maps, reference);
    }

    private void CheckFocal(string name, double focal, Reference reference)
    {
        double expected = reference.Read($"{name}_f_px", out _)[0];
        double rel = Math.Abs(focal - expected) / expected;
        output.WriteLine($"{name} f_px {focal:F4} vs {expected:F4} (rel {rel:E2})");
        Assert.True(rel <= FocalTolerance, $"{name}: f_px rel {rel:E2}");
    }

    private void CheckMaps(string name, IReadOnlyList<DepthPro2Map> maps, Reference reference)
    {
        float[] inverse = reference.Read($"{name}_inverse_depth", out long[] shape);
        float[] depth = reference.Read($"{name}_depth", out _);
        Assert.Equal(shape[0], maps.Count);
        int plane = (int)(shape[1] * shape[2]);
        double worst = 0;
        for (int t = 0; t < maps.Count; t++)
        {
            double rel = RelRms(maps[t].InverseDepth, inverse.AsSpan(t * plane, plane));
            double relDepth = RelRmsFinite(maps[t].DepthMeters, depth.AsSpan(t * plane, plane));
            output.WriteLine($"{name} frame {t}: inverse depth relRMS {rel:E2}, depth relRMS {relDepth:E2}");
            worst = Math.Max(worst, Math.Max(rel, relDepth));
        }

        Assert.True(worst <= InverseDepthTolerance, $"{name}: worst relRMS {worst:E2}");
    }

    private static SafeTensorsLoader Load(string variable)
    {
        SafeTensorsLoader loader = new();
        loader.Load(Environment.GetEnvironmentVariable(variable)!);
        return loader;
    }

    private static double RelRms(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double diff2 = 0, ref2 = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - (double)expected[i];
            diff2 += d * d;
            ref2 += (double)expected[i] * expected[i];
        }

        return Math.Sqrt(diff2 / Math.Max(ref2, 1e-30));
    }

    // The dump stores infinite depth as 0; those pixels must be infinite here too and are left out of the error.
    private static double RelRmsFinite(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double diff2 = 0, ref2 = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i] == 0f, float.IsPositiveInfinity(actual[i]));
            if (expected[i] == 0f) continue;
            double d = actual[i] - (double)expected[i];
            diff2 += d * d;
            ref2 += (double)expected[i] * expected[i];
        }

        return Math.Sqrt(diff2 / Math.Max(ref2, 1e-30));
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

        public (byte[][] Frames, int Height, int Width) Frames(int count)
        {
            int[] shape = [.. _meta.GetProperty("frames_rgb_u8").EnumerateArray().Select(e => e.GetInt32())];
            byte[] all = File.ReadAllBytes(Path.Combine(_dir, "frames_rgb_u8.bin"));
            int size = shape[1] * shape[2] * 3;
            byte[][] frames = new byte[count][];
            for (int i = 0; i < count; i++) frames[i] = all.AsSpan(i * size, size).ToArray();
            return (frames, shape[1], shape[2]);
        }

        public float[] Read(string key, out long[] shape)
        {
            shape = [.. _meta.GetProperty(key).EnumerateArray().Select(e => e.GetInt64())];
            byte[] bytes = File.ReadAllBytes(Path.Combine(_dir, key + ".bin"));
            float[] values = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }
    }
}
