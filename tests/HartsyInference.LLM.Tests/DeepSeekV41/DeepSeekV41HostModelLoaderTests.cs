using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

public sealed class DeepSeekV41HostModelLoaderTests
{
    private static readonly JsonElement Fx = DeepSeekV41ModelFixtureCheckpoint.Fx;

    private static float[] Floats(JsonElement e) => DeepSeekV41ModelFixtureCheckpoint.Floats(e);

    private static int[] Ints(JsonElement e) => DeepSeekV41ModelFixtureCheckpoint.Ints(e);

    [Fact]
    public void A_Loaded_Checkpoint_Reproduces_Upstream_Hidden_States_And_Logits()
    {
        string dir = Path.Combine(Path.GetTempPath(), "dsv41-load-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            DeepSeekV41ModelFixtureCheckpoint.Write(dir);
            using CpuBackend cpu = new();
            using DeepSeekV41LoadedModel loaded = DeepSeekV41HostModelLoader.Load(cpu, dir, new DeepSeekV41LoadOptions(MaxTokens: 64, ExpertCacheCapacity: 3));
            DeepSeekV41HostModel model = loaded.Model;
            DeepSeekV41SequenceState state = model.CreateState(64);
            int stepNo = 0;
            foreach (JsonElement step in Fx.GetProperty("steps").EnumerateArray())
            {
                int[] ids = Ints(step.GetProperty("ids"));
                float[] hidden = new float[ids.Length * model.Dim];
                model.Forward(ids, state, hidden);
                float[] expected = Floats(step.GetProperty("final"));
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(Math.Abs(expected[i] - hidden[i]) <= 1e-3f * Math.Max(1f, Math.Abs(expected[i])), $"step {stepNo} hidden[{i}]: {expected[i]} vs {hidden[i]}");
                float[] logits = model.Logits(hidden.AsSpan((ids.Length - 1) * model.Dim, model.Dim));
                float[] expectedLogits = Floats(step.GetProperty("logits"));
                for (int i = 0; i < logits.Length; i++)
                    Assert.True(Math.Abs(expectedLogits[i] - logits[i]) <= 1e-3f * Math.Max(1f, Math.Abs(expectedLogits[i])), $"step {stepNo} logits[{i}]");
                stepNo++;
            }
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public void Stored_And_Widened_Residency_Give_Identical_Hidden_States()
    {
        string dir = Path.Combine(Path.GetTempPath(), "dsv41-load-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            DeepSeekV41ModelFixtureCheckpoint.Write(dir);
            using CpuBackend cpu = new();
            float[][] results = new float[2][];
            DeepSeekV41Residency[] modes = [DeepSeekV41Residency.Stored, DeepSeekV41Residency.WidenedF32];
            for (int m = 0; m < modes.Length; m++)
            {
                using DeepSeekV41LoadedModel loaded = DeepSeekV41HostModelLoader.Load(cpu, dir, new DeepSeekV41LoadOptions(MaxTokens: 64, ExpertCacheCapacity: 3, Residency: modes[m]));
                DeepSeekV41HostModel model = loaded.Model;
                DeepSeekV41SequenceState state = model.CreateState(64);
                int[] ids = Ints(Fx.GetProperty("steps")[0].GetProperty("ids"));
                float[] hidden = new float[ids.Length * model.Dim];
                model.Forward(ids, state, hidden);
                results[m] = [.. hidden, .. model.Logits(hidden.AsSpan((ids.Length - 1) * model.Dim, model.Dim))];
            }
            Assert.Equal(results[1], results[0]);
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public void MaxLayers_Loads_Only_The_First_Layers()
    {
        string dir = Path.Combine(Path.GetTempPath(), "dsv41-load-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            DeepSeekV41ModelFixtureCheckpoint.Write(dir);
            using CpuBackend cpu = new();
            using DeepSeekV41LoadedModel loaded = DeepSeekV41HostModelLoader.Load(cpu, dir, new DeepSeekV41LoadOptions(MaxTokens: 16, MaxLayers: 1));
            Assert.Equal(1, loaded.Model.Layers);
            Assert.Throws<ArgumentOutOfRangeException>(() => new DeepSeekV41LoadOptions(8, MaxLayers: 0).Validate());
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public void Options_Must_Be_Positive()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DeepSeekV41LoadOptions(0).Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new DeepSeekV41LoadOptions(8, 0).Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new DeepSeekV41LoadOptions(8, 1, 0).Validate());
    }

    [Fact]
    public void A_Disposed_Loaded_Model_Refuses_Access_And_Disposes_Twice_Safely()
    {
        string dir = Path.Combine(Path.GetTempPath(), "dsv41-load-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            DeepSeekV41ModelFixtureCheckpoint.Write(dir);
            using CpuBackend cpu = new();
            DeepSeekV41LoadedModel loaded = DeepSeekV41HostModelLoader.Load(cpu, dir, new DeepSeekV41LoadOptions(MaxTokens: 16));
            Assert.NotNull(loaded.Model);
            loaded.Dispose();
            loaded.Dispose();
            Assert.Throws<ObjectDisposedException>(() => loaded.Model);
            Assert.Throws<ObjectDisposedException>(() => loaded.Checkpoint);
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }
}
