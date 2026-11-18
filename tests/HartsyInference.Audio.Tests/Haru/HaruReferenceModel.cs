using HartsyInference.Audio.Models.Haru;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests.Haru;

/// <summary>A Haru checkpoint opened on the CPU backend, and the comparisons with the upstream PyTorch outputs that the tiny-checkpoint and
/// the released-weights tests share. The expected files follow the layout of <c>tools/haru/*reference.py</c>.</summary>
internal sealed unsafe class HaruReferenceModel : IDisposable
{
    private readonly SafeTensorsLoader _weights = new();

    internal readonly HaruConfig Config;
    internal readonly Dictionary<string, Tensor> Tensors;
    internal readonly CpuBackend Backend = new();
    internal readonly HaruLm Lm;
    internal readonly HaruMeanFlowHead Head;
    internal readonly HaruCodecSession Codec;

    internal HaruReferenceModel(string weightsPath, string configPath, int capacity)
    {
        _weights.Load(weightsPath);
        Config = HaruConfig.FromJson(File.ReadAllText(configPath));
        Tensors = _weights.GetAllTensors();
        Lm = new HaruLm(Config, Tensors, capacity);
        Head = new HaruMeanFlowHead(Config.Head);
        Head.LoadWeights(Tensors);
        Codec = new HaruCodecSession(HaruCodec.Create(Config.Codec, Tensors), Backend, Config.LatentDim, Config.Codec.FixStd);
    }

    internal static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    internal static int[] ReadInts(Tensor t) => new ReadOnlySpan<int>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    internal static Tensor Make(float[] values, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        values.AsSpan().CopyTo(new Span<float>((void*)t.DataPointer, values.Length));
        return t;
    }

    /// <summary>Largest absolute difference over the largest absolute reference value.</summary>
    internal static float RelativeError(ReadOnlySpan<float> got, ReadOnlySpan<float> want)
    {
        Assert.Equal(want.Length, got.Length);
        float worst = 0;
        float scale = 1e-12f;
        for (int i = 0; i < got.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(got[i] - want[i]));
            scale = MathF.Max(scale, MathF.Abs(want[i]));
        }
        return worst / scale;
    }

    /// <summary>The prompt in one call, then single positions: relative errors of the hidden states, the text logits of the last steps and
    /// the F32 PAD/EOP logits.</summary>
    internal (float Hidden, float Logits, float Stream) CompareLanguageModel(string expectedPath)
    {
        using SafeTensorsLoader expected = new();
        expected.Load(expectedPath);
        int[] text = ReadInts(expected.GetTensor("in.text"));
        float[] modelLatents = Read(expected.GetTensor("in.model_latents"));
        float[] userLatents = Read(expected.GetTensor("in.user_latents"));
        bool[] modelMask = [.. ReadInts(expected.GetTensor("in.model_mask")).Select(static m => m != 0)];
        bool[] userMask = [.. ReadInts(expected.GetTensor("in.user_mask")).Select(static m => m != 0)];
        int prefill = ReadInts(expected.GetTensor("in.prefill"))[0];
        int total = text.Length;
        int d = Config.LatentDim;
        int h = Config.Lm.HiddenSize;
        float[] hidden = new float[total * h];
        float[] logits = new float[(total - prefill) * Config.Lm.VocabSize];
        float[] stream = new float[(total - prefill) * 2];
        int[] streamTokens = [Config.PadTokenId, Config.EopTokenId];
        Lm.Reset();
        List<(int Start, int End)> spans = [(0, prefill)];
        for (int i = prefill; i < total; i++)
        {
            spans.Add((i, i + 1));
        }
        foreach ((int start, int end) in spans)
        {
            int rows = end - start;
            using Tensor embeds = Lm.Embed(Backend, text.AsSpan(start, rows), modelLatents.AsSpan(start * d, rows * d), modelMask.AsSpan(start, rows),
                userLatents.AsSpan(start * d, rows * d), userMask.AsSpan(start, rows));
            using Tensor states = Lm.Forward(Backend, embeds, rows);
            Read(states).CopyTo(hidden, start * h);
            if (start >= prefill)
            {
                int step = start - prefill;
                using Tensor row = Lm.Logits(Backend, states, rows);
                Read(row).CopyTo(logits, step * Config.Lm.VocabSize);
                Lm.StreamLogits(states, rows - 1, streamTokens, stream.AsSpan(step * 2, 2));
            }
        }
        Assert.Equal(total, Lm.Position);
        return (RelativeError(hidden, Read(expected.GetTensor("ref.hidden"))), RelativeError(logits, Read(expected.GetTensor("ref.logits"))),
            RelativeError(stream, Read(expected.GetTensor("ref.stream_logits"))));
    }

    /// <summary>The MeanFlow head on the reference hidden states and noise, un-normalized as the session does.</summary>
    internal float CompareHead(string expectedPath)
    {
        using SafeTensorsLoader expected = new();
        expected.Load(expectedPath);
        float[] hidden = Read(expected.GetTensor("ref.hidden"));
        float[] noise = Read(expected.GetTensor("in.noise"));
        int n = hidden.Length / Config.Lm.HiddenSize;
        using Tensor h = Make(hidden, 1, n, Config.Lm.HiddenSize);
        using Tensor z = Make(noise, 1, n, Config.LatentDim);
        using Tensor normalized = Head.Sample(Backend, h, z);
        float[] latents = Read(normalized);
        for (int i = 0; i < latents.Length; i++)
        {
            latents[i] = Lm.Unscale(latents[i]);
        }
        return RelativeError(latents, Read(expected.GetTensor("ref.head_latents")));
    }

    /// <summary>A recorded upstream conversation replayed: <paramref name="prefix"/> names its tensors (<c>{prefix}.tokens</c>, <c>.audio</c>,
    /// <c>.draw.N</c>). Returns whether tokens and text are identical, the audio error, and the draws left over.</summary>
    internal (bool TokensEqual, bool TextEqual, float Audio, int LeftoverDraws, int[] Tokens) ReplayConversation(string expectedPath, string prefix, string card,
        string? referenceKey, string[] wantText, HaruTextCodec text, HaruDuplexOptions options)
    {
        using SafeTensorsLoader expected = new();
        expected.Load(expectedPath);
        List<float[]> draws = [];
        for (int i = 0; expected.Descriptors.ContainsKey($"{prefix}.draw.{i}"); i++)
        {
            draws.Add(Read(expected.GetTensor($"{prefix}.draw.{i}")));
        }
        HaruScriptedNoise noise = new(draws);
        HaruDuplexSession session = new(Config, Lm, Head, Codec, text, Backend, options, noise);
        float[] user = Read(expected.GetTensor("in.user"));
        float[] reference = referenceKey is null ? [] : Read(expected.GetTensor(referenceKey));
        int[] wantTokens = ReadInts(expected.GetTensor($"{prefix}.tokens"));
        float[] wantAudio = Read(expected.GetTensor($"{prefix}.audio"));
        int frame = Config.FrameSamples;
        List<HaruFrame> frames = [session.Start(card, reference)];
        for (int i = 0; i < wantTokens.Length - 1; i++)
        {
            frames.Add(session.Step(user.AsSpan(i * frame, frame)));
        }
        float[] audio = new float[wantAudio.Length];
        for (int f = 0; f < frames.Count; f++)
        {
            frames[f].Audio.CopyTo(audio, f * frame);
        }
        int[] tokens = [.. frames.Select(static f => f.Token)];
        return (tokens.SequenceEqual(wantTokens), frames.Select(static f => f.Text).SequenceEqual(wantText), RelativeError(audio, wantAudio),
            noise.Remaining, tokens);
    }

    public void Dispose()
    {
        Codec.Dispose();
        Head.Dispose();
        Lm.Dispose();
        Backend.Dispose();
        _weights.Dispose();
    }
}
