using System.Text.Json;
using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight check of the video-conditioning path. <c>HARTSY_CONTROLFOLEY_VIDEO_DIR</c> must hold
/// <c>synchformer_state_dict.pth</c>, <c>cav_mae_st.pth</c> and <c>video_real_ref.safetensors/.json</c>
/// (<c>tools/controlfoley/video_reference.py real</c>); the test does nothing when it is unset.</summary>
[Trait("Category", "Integration")]
public sealed unsafe class ControlFoleyVideoRealWeightTests(ITestOutputHelper output)
{
    private static string? Dir => Environment.GetEnvironmentVariable("HARTSY_CONTROLFOLEY_VIDEO_DIR");

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    private static (float Diff, float Scale) Compare(float[] want, float[] got)
    {
        Assert.Equal(want.Length, got.Length);
        float worst = 0f, scale = 0f;
        for (int i = 0; i < want.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(want[i] - got[i]));
            scale = MathF.Max(scale, MathF.Abs(want[i]));
        }

        return (worst, scale);
    }

    [Fact]
    public void VideoPath_MatchesPythonOnRealWeightsAndClip()
    {
        if (Dir is not { Length: > 0 } dir)
        {
            return;
        }

        using SafeTensorsLoader refLoader = new();
        refLoader.Load(Path.Combine(dir, "video_real_ref.safetensors"));
        IReadOnlyDictionary<string, Tensor> r = refLoader.GetAllTensors();
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "video_real_ref.json")));

        Tensor frameTensor = r["in.frames"];
        int n = (int)frameTensor.Shape[0], h = (int)frameTensor.Shape[1], w = (int)frameTensor.Shape[2];
        byte[][] frames = new byte[n][];
        for (int i = 0; i < n; i++)
        {
            frames[i] = new ReadOnlySpan<byte>((byte*)frameTensor.DataPointer + (long)i * h * w * 3, h * w * 3).ToArray();
        }

        ControlFoleyRawVideo video = new()
        {
            Frames = frames, Times = new ReadOnlySpan<double>((void*)r["in.times"].DataPointer, n).ToArray(), Width = w, Height = h,
        };
        ControlFoleyVideoInput input = ControlFoleyVideoPreprocessor.Prepare(video, doc.RootElement.GetProperty("duration").GetDouble());
        Assert.Equal(doc.RootElement.GetProperty("total_duration").GetDouble(), input.TotalDuration, 9);
        foreach ((string name, float[] got) in new[] { ("clip", input.Clip), ("visual", input.Visual), ("sync_frames", input.Sync) })
        {
            (float d, float m) = Compare(Read(r["ref." + name]), got);
            output.WriteLine($"{name} frames max |d| = {d:E2} (max |ref| {m:F3})");
            Assert.True(d <= 1e-5f, $"{name} preprocessing differs by {d}");
        }

        using IBackend backend = new CpuBackend();
        using AnyFormatCheckpointLoader syncLoader = new();
        syncLoader.Load(Path.Combine(dir, "synchformer_state_dict.pth"));
        using ControlFoleySynchformer synchformer = new(ControlFoleySynchformerConfig.Released);
        synchformer.LoadWeights(syncLoader.GetAllTensors());
        (float syncDiff, float syncScale) = Compare(Read(r["ref.sync"]), synchformer.Encode(backend, input.Sync));
        output.WriteLine($"synchformer max |d| = {syncDiff:E2} (max |ref| {syncScale:F3})");

        using AnyFormatCheckpointLoader cavLoader = new();
        cavLoader.Load(Path.Combine(dir, "cav_mae_st.pth"));
        using ControlFoleyCavMae cav = new(ControlFoleyCavMaeConfig.Released);
        cav.LoadWeights(cavLoader.GetAllTensors());
        (float cavDiff, float cavScale) = Compare(Read(r["ref.cav_pooled"]), cav.EncodePooled(backend, input.Visual));
        output.WriteLine($"cav-mae pooled max |d| = {cavDiff:E2} (max |ref| {cavScale:F3})");

        Assert.True(syncDiff <= 1e-3f * Math.Max(1f, syncScale), $"synchformer differs by {syncDiff}");
        Assert.True(cavDiff <= 1e-3f * Math.Max(1f, cavScale), $"cav-mae differs by {cavDiff}");
    }
}
