using System.Text.Json;
using HartsyInference.Cli.Commands;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary><c>hartsy vision --mode metric-depth</c>: the mode a Depth Pro 2 id implies, the depth preview of an infinite pixel, and (with
/// <c>HARTSY_DEPTHPRO2_WEIGHTS</c> and <c>HARTSY_DEPTHPRO2_FOCAL_WEIGHTS</c> set) a single image, a folder of views and a folder of video frames (buffered and
/// unbuffered) through the CLI dispatch on CPU.</summary>
public sealed class DepthPro2CliTests : IDisposable
{
    private const string WeightsVariable = "HARTSY_DEPTHPRO2_WEIGHTS";
    private const string FocalVariable = "HARTSY_DEPTHPRO2_FOCAL_WEIGHTS";
    private const int Width = 100, Height = 70;

    private readonly string _folder = Directory.CreateTempSubdirectory("hartsy-depthpro2-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    [Theory]
    [InlineData("depthpro2-s", "metric-depth")]
    [InlineData("DepthPro2-H", "metric-depth")]
    [InlineData("depth-anything", "depth")]
    public void TheModelIdImpliesTheMode(string model, string mode) => Assert.Equal(mode, VisionCommand.InferMode(model));

    [Fact]
    public void AModelPathToADepthPro2FileIsFoundByItsTensors()
    {
        string depth = Path.Combine(_folder, "dp2.safetensors"), other = Path.Combine(_folder, "other.safetensors");
        using (Tensor qkv = new(new TensorShape(3 * 384, 384), DType.F32))
        using (Tensor head = new(new TensorShape(1, 32, 1, 1), DType.F32))
        {
            SafeTensorsWriter.Save(depth, new Dictionary<string, Tensor> { ["encoder.blocks.0.attn.qkv.weight"] = qkv, ["head.4.weight"] = head });
            SafeTensorsWriter.Save(other, new Dictionary<string, Tensor> { ["trunk.blocks.0.attn.qkv.weight"] = qkv });
        }

        Assert.True(VisionCommand.IsDepthPro2File(depth));
        Assert.False(VisionCommand.IsDepthPro2File(other));
        Assert.False(VisionCommand.IsDepthPro2File(Path.Combine(_folder, "missing.safetensors")));
        Assert.False(VisionCommand.IsDepthPro2File(null));
    }

    [Fact]
    public void TheDepthPreviewDrawsInfiniteDepthAsTheFarEnd()
    {
        VisionResult result = new()
        {
            DepthMap = new VisionDepthMap
            {
                Values = new VisionArray([1f, 3f, float.PositiveInfinity, 2f], 2, 2), Kind = VisionDepthKind.MetricMeters,
                ValidMask = new VisionArray([1f, 1f, 0f, 1f], 2, 2),
            },
        };

        VisionOutputWriter.WriteResult(result, _folder);

        (byte[] rgb, _, _) = ImageIo.DecodeFile(Path.Combine(_folder, "depth.png"));
        Assert.Equal([0, 255, 255, 128], new[] { rgb[0], rgb[3], rgb[6], rgb[9] });
        Assert.True(File.Exists(Path.Combine(_folder, "depth-valid.npy")));
    }

    [RealWeightsFact(WeightsVariable, FocalVariable)]
    public async Task ASingleImageGivesMetricDepthAndTheFocalItUsed()
    {
        string image = WriteScene("scene.png", mirror: false);
        using InferenceEngine engine = new("cpu");

        string estimated = await RunAsync(engine, image, focalPx: null);
        string given = await RunAsync(engine, image, focalPx: 500);

        double focal = Camera(estimated, "").GetProperty("fx").GetDouble();
        Assert.InRange(focal, 10, 100_000);
        JsonElement camera = Camera(given, "");
        Assert.Equal(500, camera.GetProperty("fx").GetDouble());
        Assert.Equal(Width / 2.0, camera.GetProperty("cx").GetDouble());
        Assert.Equal(Height / 2.0, camera.GetProperty("cy").GetDouble());
        // Depth scales with the focal length: f / (W Â· inverse depth).
        float[] a = Depth(estimated, ""), b = Depth(given, "");
        for (int i = 0; i < a.Length; i++)
        {
            if (float.IsPositiveInfinity(a[i])) continue;
            Assert.Equal(a[i] * 500 / focal, b[i], a[i] * 1e-4);
        }
    }

    [RealWeightsFact(WeightsVariable, FocalVariable)]
    public async Task AFolderOfViewsGivesOneDepthPerViewWithOneSharedFocal()
    {
        string views = Directory.CreateDirectory(Path.Combine(_folder, "views")).FullName;
        File.Move(WriteScene("v0.png", mirror: false), Path.Combine(views, "v0.png"));
        File.Move(WriteScene("v1.png", mirror: true), Path.Combine(views, "v1.png"));
        using InferenceEngine engine = new("cpu");

        string dir = await RunAsync(engine, views, focalPx: null);

        double f0 = Camera(dir, "view-00.").GetProperty("fx").GetDouble(), f1 = Camera(dir, "view-01.").GetProperty("fx").GetDouble();
        Assert.Equal(f0, f1);
        Assert.Equal(Width * Height, Depth(dir, "view-00.").Length);
        Assert.Equal(Width * Height, Depth(dir, "view-01.").Length);
    }

    [RealWeightsFact(WeightsVariable, FocalVariable)]
    public async Task AFolderWithFpsIsAVideoWithOneDepthPerFrameBufferedOrNot()
    {
        string clip = Directory.CreateDirectory(Path.Combine(_folder, "clip")).FullName;
        for (int i = 0; i < 5; i++) File.Move(WriteScene($"f{i}.png", mirror: false, shift: 4 * i), Path.Combine(clip, $"f{i}.png"));
        using InferenceEngine engine = new("cpu");

        foreach (string[] options in new[] { new[] { "window", "3", "overlap", "1" }, new[] { "window", "3", "unbuffered", "true" } })
        {
            string dir = await RunAsync(engine, clip, focalPx: null, options);

            using JsonDocument index = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "frames.json")));
            JsonElement[] frames = [.. index.RootElement.EnumerateArray()];
            Assert.Equal(5, frames.Length);
            double focal = Camera(dir, "frame-00000.").GetProperty("fx").GetDouble();
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(i, frames[i].GetProperty("index").GetInt32());
                Assert.Equal(i / 10.0, frames[i].GetProperty("timestampSeconds").GetDouble(), 1e-9);
                Assert.Equal(focal, Camera(dir, $"frame-{i:D5}.").GetProperty("fx").GetDouble());
                Assert.Equal(Width * Height, Depth(dir, $"frame-{i:D5}.").Length);
            }
        }
    }

    private async Task<string> RunAsync(InferenceEngine engine, string input, double? focalPx, string[]? video = null)
    {
        ModelSpec spec = ModelResolver.Resolve("depthpro2-s", Environment.GetEnvironmentVariable(WeightsVariable), Modality.Vision) with
        {
            Aux = new Dictionary<string, string> { [VisionService.DepthPro2FocalAuxKey] = Environment.GetEnvironmentVariable(FocalVariable)! },
        };
        ParamState parameters = new(Modality.Vision);
        parameters.Put("mode", VisionCommand.InferMode("depthpro2-s"));
        if (focalPx is double f) parameters.Put("focal-px", f.ToString(System.Globalization.CultureInfo.InvariantCulture));
        if (video is not null)
        {
            parameters.Put("fps", "10");
            for (int i = 0; i < video.Length; i += 2) parameters.Put(video[i], video[i + 1]);
        }
        string output = Directory.CreateDirectory(Path.Combine(_folder, "run-" + Guid.NewGuid().ToString("N"))).FullName;

        await GenerationDispatch.RunAsync(engine, spec, input, parameters, output, quiet: true, CancellationToken.None);

        return Directory.GetDirectories(output).Single();
    }

    private static JsonElement Camera(string dir, string prefix) =>
        JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, prefix + "result.json"))).RootElement.GetProperty("camera");

    private static float[] Depth(string dir, string prefix)
    {
        VisionArray depth = VisionArrayCodec.FromNpy(File.ReadAllBytes(Path.Combine(dir, prefix + "depth.npy")));
        Assert.Equal([Height, Width], depth.Shape);
        Assert.All(depth.Data, v => Assert.True(v > 0));
        return depth.Data;
    }

    /// <summary>A floor-and-wall scene: a horizontal gradient wall above a textured floor that darkens towards the bottom, with a box.</summary>
    private string WriteScene(string name, bool mirror, int shift = 0)
    {
        byte[] rgb = new byte[Width * Height * 3];
        for (int y = 0; y < Height; y++)
            for (int x = 0; x < Width; x++)
            {
                int sx = (mirror ? Width - 1 - x : x) + shift, p = (y * Width + x) * 3;
                bool floor = y > Height / 2, box = sx > 30 && sx < 55 && y > 25 && y < 50;
                rgb[p] = (byte)(box ? 200 : floor ? 90 + (sx * 7 + y * 3) % 40 : 150 + sx / 2);
                rgb[p + 1] = (byte)(box ? 60 : floor ? 70 + y : 170);
                rgb[p + 2] = (byte)(box ? 50 : floor ? 40 : 210 - y);
            }

        string path = Path.Combine(_folder, name);
        File.WriteAllBytes(path, PngEncoder.Encode(rgb, Width, Height));
        return path;
    }
}
