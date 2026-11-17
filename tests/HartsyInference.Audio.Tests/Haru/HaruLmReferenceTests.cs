using HartsyInference.Audio.Models.Haru;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Haru;

/// <summary>Haru's language model (stream-summed input, Qwen3.5 trunk, text logits, F32 PAD/EOP logits) against the upstream PyTorch
/// model (<c>tools/haru/haru_reference.py</c>, CPU F32) on a tiny random checkpoint: the prompt in one call, then single positions.</summary>
public sealed unsafe class HaruLmReferenceTests(ITestOutputHelper output)
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Haru");

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    private static int[] ReadInts(Tensor t) => new ReadOnlySpan<int>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    private static float RelativeError(string name, ReadOnlySpan<float> got, ReadOnlySpan<float> want, ITestOutputHelper output)
    {
        Assert.Equal(want.Length, got.Length);
        float worst = 0;
        float scale = 1e-12f;
        for (int i = 0; i < got.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(got[i] - want[i]));
            scale = MathF.Max(scale, MathF.Abs(want[i]));
        }
        output.WriteLine($"{name}: n={got.Length} max|d|={worst:E3} max|ref|={scale:E3} rel={worst / scale:E3}");
        return worst / scale;
    }

    [Fact]
    public void Prompt_AndSingleSteps_MatchUpstream()
    {
        using SafeTensorsLoader weights = new();
        weights.Load(Path.Combine(Dir, "haru_tiny.safetensors"));
        using SafeTensorsLoader expected = new();
        expected.Load(Path.Combine(Dir, "haru_tiny_expected.safetensors"));
        HaruConfig config = HaruConfig.FromJson(File.ReadAllText(Path.Combine(Dir, "config.json")));
        Dictionary<string, Tensor> tensors = weights.GetAllTensors();
        using HaruLm lm = new(config, tensors, capacity: 64);
        using CpuBackend backend = new();

        int[] text = ReadInts(expected.GetTensor("in.text"));
        float[] modelLatents = Read(expected.GetTensor("in.model_latents"));
        float[] userLatents = Read(expected.GetTensor("in.user_latents"));
        bool[] modelMask = [.. ReadInts(expected.GetTensor("in.model_mask")).Select(static m => m != 0)];
        bool[] userMask = [.. ReadInts(expected.GetTensor("in.user_mask")).Select(static m => m != 0)];
        int prefill = ReadInts(expected.GetTensor("in.prefill"))[0];
        int total = text.Length;
        int d = config.LatentDim;
        int h = config.Lm.HiddenSize;

        float[] hidden = new float[total * h];
        float[] logits = new float[(total - prefill) * config.Lm.VocabSize];
        float[] streamLogits = new float[(total - prefill) * 2];
        int[] streamTokens = [config.PadTokenId, config.EopTokenId];
        List<(int Start, int End)> spans = [(0, prefill)];
        for (int i = prefill; i < total; i++)
        {
            spans.Add((i, i + 1));
        }
        foreach ((int start, int end) in spans)
        {
            int rows = end - start;
            using Tensor embeds = lm.Embed(backend, text.AsSpan(start, rows), modelLatents.AsSpan(start * d, rows * d), modelMask.AsSpan(start, rows),
                userLatents.AsSpan(start * d, rows * d), userMask.AsSpan(start, rows));
            using Tensor states = lm.Forward(backend, embeds, rows);
            Read(states).CopyTo(hidden, start * h);
            if (start >= prefill)
            {
                int step = start - prefill;
                using Tensor row = lm.Logits(backend, states, rows);
                Read(row).CopyTo(logits, step * config.Lm.VocabSize);
                lm.StreamLogits(states, rows - 1, streamTokens, streamLogits.AsSpan(step * 2, 2));
            }
        }
        Assert.Equal(total, lm.Position);

        float hiddenError = RelativeError("hidden states (22 prefill + 6 steps)", hidden, Read(expected.GetTensor("ref.hidden")), output);
        float logitError = RelativeError("text logits of the 6 steps", logits, Read(expected.GetTensor("ref.logits")), output);
        float streamError = RelativeError("PAD/EOP logits (F32)", streamLogits, Read(expected.GetTensor("ref.stream_logits")), output);
        Assert.True(hiddenError < 5e-5f, $"hidden states: relative error {hiddenError:E3}");
        Assert.True(logitError < 5e-5f, $"text logits: relative error {logitError:E3}");
        Assert.True(streamError < 5e-5f, $"PAD/EOP logits: relative error {streamError:E3}");
    }

    [Fact]
    public void Reset_RestartsTheConversation()
    {
        using SafeTensorsLoader weights = new();
        weights.Load(Path.Combine(Dir, "haru_tiny.safetensors"));
        HaruConfig config = HaruConfig.FromJson(File.ReadAllText(Path.Combine(Dir, "config.json")));
        Dictionary<string, Tensor> tensors = weights.GetAllTensors();
        using HaruLm lm = new(config, tensors, capacity: 64);
        using CpuBackend backend = new();
        int[] tokens = [3, 4, 5];
        using Tensor embeds = lm.Embed(backend, tokens, [], [], [], []);
        using Tensor first = lm.Forward(backend, embeds, 3);
        float[] before = Read(first);
        Assert.Equal(3, lm.Position);
        lm.Reset();
        Assert.Equal(0, lm.Position);
        using Tensor again = lm.Forward(backend, embeds, 3);
        Assert.Equal(before, Read(again));
    }

    [Fact]
    public void Embed_RefusesLatentsOfTheWrongLength()
    {
        using SafeTensorsLoader weights = new();
        weights.Load(Path.Combine(Dir, "haru_tiny.safetensors"));
        HaruConfig config = HaruConfig.FromJson(File.ReadAllText(Path.Combine(Dir, "config.json")));
        Dictionary<string, Tensor> tensors = weights.GetAllTensors();
        using HaruLm lm = new(config, tensors, capacity: 8);
        using CpuBackend backend = new();
        Assert.Throws<ArgumentException>(() => lm.Embed(backend, [1, 2], new float[3], [], [], []));
        Assert.Throws<ArgumentException>(() => lm.Embed(backend, [], [], [], [], []));
    }
}
