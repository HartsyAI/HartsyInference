using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Decision.Clef;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>A tiny random joint schema head run through the official <c>JointSchemaHead</c>
/// (<c>tools/clef/head_reference.py</c>) must match the port's per-option logits.</summary>
public sealed unsafe class ClefJointHeadParityTests
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Clef");

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    [Fact]
    public void Logits_MatchOfficialHead()
    {
        using SafeTensorsLoader headLoader = new();
        headLoader.Load(Path.Combine(Dir, "clef_head_tiny.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in headLoader.Descriptors.Keys)
        {
            weights[name] = headLoader.GetTensor(name);
        }
        using SafeTensorsLoader inputLoader = new();
        inputLoader.Load(Path.Combine(Dir, "clef_head_inputs.safetensors"));
        float[] hidden = Read(inputLoader.GetTensor("hidden"));
        float[] lmHead = Read(inputLoader.GetTensor("lm_head"));
        Tensor idTensor = inputLoader.GetTensor("input_ids");
        int[] ids = new ReadOnlySpan<int>((void*)idTensor.DataPointer, (int)idTensor.ElementCount).ToArray();
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "clef_head_expected.json")));
        JsonElement cfg = doc.RootElement.GetProperty("config");
        ClefJointHeadConfig config = new()
        {
            HiddenSize = cfg.GetProperty("hidden_size").GetInt32(), Width = cfg.GetProperty("width").GetInt32(),
            RoutingLayers = cfg.GetProperty("routing_layers").GetInt32(), Layers = cfg.GetProperty("layers").GetInt32(),
            Heads = cfg.GetProperty("heads").GetInt32(), Feedforward = cfg.GetProperty("feedforward").GetInt32(),
        };
        List<ClefQuestionSpans> questions = [];
        List<float[]> want = [];
        foreach (JsonElement q in doc.RootElement.GetProperty("questions").EnumerateArray())
        {
            JsonElement span = q.GetProperty("question");
            questions.Add(new ClefQuestionSpans
            {
                QuestionId = q.GetProperty("id").GetString()!, QuestionType = q.GetProperty("type").GetInt32(),
                QuestionSpan = (span[0].GetInt32(), span[1].GetInt32()),
                OptionSpans = q.GetProperty("options").EnumerateArray().Select(s => (s[0].GetInt32(), s[1].GetInt32())).ToArray(),
                OptionIds = q.GetProperty("optionIds").EnumerateArray().Select(s => s.GetString()!).ToArray(),
            });
        }
        foreach (JsonElement l in doc.RootElement.GetProperty("logits").EnumerateArray())
        {
            want.Add(l.EnumerateArray().Select(v => v.GetSingle()).ToArray());
        }

        using CpuBackend backend = new();
        using ClefJointHead head = new(config);
        head.LoadWeights(weights);
        float[][] got = head.Forward(backend, hidden, doc.RootElement.GetProperty("seq").GetInt32(), ids, questions,
            id => lmHead.AsSpan(id * config.HiddenSize, config.HiddenSize).ToArray());

        Assert.Equal(want.Count, got.Length);
        for (int i = 0; i < want.Count; i++)
        {
            Assert.Equal(want[i].Length, got[i].Length);
            for (int j = 0; j < want[i].Length; j++)
            {
                Assert.True(MathF.Abs(want[i][j] - got[i][j]) < 1e-4f, $"question {i} option {j}: {want[i][j]} vs {got[i][j]}");
            }
        }
    }
}
