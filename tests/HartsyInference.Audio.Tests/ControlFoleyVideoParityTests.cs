using System.Text.Json;
using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tiny random official Synchformer / CAV-MAE-ST instances and a synthetic mp4 run through the official frame
/// loader (<c>tools/controlfoley/video_reference.py tiny</c>) must match the port.</summary>
public sealed unsafe class ControlFoleyVideoParityTests
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "ControlFoleyVideo");

    private static Dictionary<string, Tensor> Open(string name, out SafeTensorsLoader loader)
    {
        loader = new SafeTensorsLoader();
        loader.Load(Path.Combine(Dir, name));
        SafeTensorsLoader l = loader;
        return l.Descriptors.Keys.ToDictionary(k => k, l.GetTensor);
    }

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    private static float MaxAbs(float[] want, float[] got)
    {
        Assert.Equal(want.Length, got.Length);
        float worst = 0f;
        for (int i = 0; i < want.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(want[i] - got[i]));
        }

        return worst;
    }

    [Fact]
    public void Synchformer_MatchesOfficialEncodeVideoWithSync()
    {
        Dictionary<string, Tensor> w = Open("video_sync_tiny.safetensors", out SafeTensorsLoader loader);
        using (loader)
        {
            float[] seed = Read(w["in.seed"]);
            float[] frames = new float[32 * 3 * 224 * 224];
            for (int f = 0; f < 32 * 3; f++)
            {
                for (int y = 0; y < 224; y++)
                {
                    for (int x = 0; x < 224; x++)
                    {
                        frames[(f * 224 + y) * 224 + x] = seed[(f * 14 + y / 16) * 14 + x / 16];
                    }
                }
            }

            using IBackend backend = new CpuBackend();
            using ControlFoleySynchformer model = new(new ControlFoleySynchformerConfig { EmbedDim = 48, Depth = 2, Heads = 3, PatchSize = 56 });
            model.LoadWeights(w);
            float[] got = model.Encode(backend, frames);
            float diff = MaxAbs(Read(w["ref.sync"]), got);
            Assert.True(diff <= 1e-4f, $"sync max |d| = {diff}");
        }
    }

    [Fact]
    public void CavMaeVisualBranch_MatchesOfficialForwardFeatV()
    {
        Dictionary<string, Tensor> w = Open("video_cav_tiny.safetensors", out SafeTensorsLoader loader);
        using (loader)
        {
            ControlFoleyCavMaeConfig cfg = new() { EmbedDim = 48, Heads = 3, PatchSize = 16, InputSize = 64, ModalitySpecificDepth = 2 };
            using IBackend backend = new CpuBackend();
            using ControlFoleyCavMae model = new(cfg);
            model.LoadWeights(w);
            float[] frames = Read(w["in.frames"]);
            float tokens = MaxAbs(Read(w["ref.tokens"]), model.EncodeFrames(backend, frames));
            float pooled = MaxAbs(Read(w["ref.pooled"]), model.EncodePooled(backend, frames));
            Assert.True(tokens <= 1e-4f, $"tokens max |d| = {tokens}");
            Assert.True(pooled <= 1e-4f, $"pooled max |d| = {pooled}");
        }
    }

    [Fact]
    public void FramePreprocessing_MatchesOfficialLoadVideo()
    {
        Dictionary<string, Tensor> w = Open("video_frames_tiny.safetensors", out SafeTensorsLoader loader);
        using (loader)
        using (JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "video_frames_tiny.json"))))
        {
            Tensor frameTensor = w["in.frames"];
            int n = (int)frameTensor.Shape[0], h = (int)frameTensor.Shape[1], width = (int)frameTensor.Shape[2];
            byte[][] frames = new byte[n][];
            for (int i = 0; i < n; i++)
            {
                frames[i] = new ReadOnlySpan<byte>((byte*)frameTensor.DataPointer + (long)i * h * width * 3, h * width * 3).ToArray();
            }

            double[] times = new ReadOnlySpan<double>((void*)w["in.times"].DataPointer, n).ToArray();
            ControlFoleyRawVideo video = new() { Frames = frames, Times = times, Width = width, Height = h };
            JsonElement sizes = doc.RootElement.GetProperty("sizes");
            ControlFoleyVideoOptions options = new()
            {
                ClipSize = sizes.GetProperty("clip").GetInt32(), VisualSize = sizes.GetProperty("visual").GetInt32(),
                SyncSize = sizes.GetProperty("sync").GetInt32(),
            };
            int index = 0;
            foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
            {
                ControlFoleyVideoInput got = ControlFoleyVideoPreprocessor.Prepare(video, c.GetProperty("duration").GetDouble(), options);
                Assert.Equal(c.GetProperty("total_duration").GetDouble(), got.TotalDuration, 9);
                // One uint8 level is 1/255/std (<= 0.9 after normalisation); the resize must be exact, so the tolerance is float noise.
                float clip = MaxAbs(Read(w[$"ref.{index}.clip"]), got.Clip);
                float visual = MaxAbs(Read(w[$"ref.{index}.visual"]), got.Visual);
                float sync = MaxAbs(Read(w[$"ref.{index}.sync"]), got.Sync);
                Assert.True(clip <= 1e-5f && visual <= 1e-5f && sync <= 1e-5f, $"case {index}: clip {clip}, visual {visual}, sync {sync}");
                index++;
            }
        }
    }
}
