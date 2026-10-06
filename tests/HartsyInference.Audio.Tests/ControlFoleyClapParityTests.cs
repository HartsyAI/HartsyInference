using System.Text.Json;
using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>A tiny random laion_clap HTS-AT tower (<c>tools/controlfoley/clap_reference.py tiny</c>) run through the official
/// <c>get_audio_features</c> / <c>get_audio_embedding</c> path must match the port, including the repeat padding, the log-mel and
/// the bicubic stretch of the 1001-frame mel to the 1024-frame Swin image.</summary>
public sealed unsafe class ControlFoleyClapParityTests
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "ControlFoleyClap");

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    internal static float MaxAbs(float[] want, float[] got)
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
    public void AudioEmbedding_MatchesOfficialImplementation()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(Dir, "clap_tiny.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in loader.Descriptors.Keys)
        {
            weights[name] = loader.GetTensor(name);
        }

        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "clap_tiny.json")));
        JsonElement root = doc.RootElement;
        ControlFoleyClapConfig config = new()
        {
            EmbedDim = root.GetProperty("embed_dim").GetInt32(),
            Depths = root.GetProperty("depths").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            Heads = root.GetProperty("heads").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            JointDim = root.GetProperty("joint").GetInt32(),
        };

        using IBackend backend = new CpuBackend();
        ControlFoleyClap clap = new(config);
        clap.LoadWeights(weights);
        for (int i = 0; i < root.GetProperty("cases").GetInt32(); i++)
        {
            float[] audio = Read(weights[$"in.audio{i}"]);
            float melDiff = MaxAbs(Read(weights[$"ref.logmel{i}"]), clap.LogMel(audio));
            Assert.True(melDiff <= 2e-3f, $"case {i}: log-mel max |d| = {melDiff}");
            float[] embedding = clap.Embed(backend, audio);
            float diff = MaxAbs(Read(weights[$"ref.embedding{i}"]), embedding);
            Assert.True(diff <= 1e-5f, $"case {i}: embedding max |d| = {diff}");
        }
    }
}
