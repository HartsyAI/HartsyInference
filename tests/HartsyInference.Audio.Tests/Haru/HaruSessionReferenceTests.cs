using System.Text.Json;
using HartsyInference.Audio.Models.Haru;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Haru;

/// <summary>The whole conversation loop (prompt, prefill, frame by frame user audio, text and audio out) against the upstream
/// <c>DuplexSession</c> (<c>tools/haru/haru_session_reference.py</c>, CPU F32, greedy) on the tiny checkpoint. Every random draw of the
/// upstream run is recorded and replayed, so tokens, text and audio can be compared exactly.</summary>
public sealed unsafe class HaruSessionReferenceTests(ITestOutputHelper output)
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Haru");

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    private static int[] ReadInts(Tensor t) => new ReadOnlySpan<int>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    // Byte tokenizer of the reference script: the UTF-8 bytes of the text are the ids; id 7 decodes to U+FFFD, any other id to chr(33 + id % 90).
    private static readonly HaruTextCodec ByteCodec = new(
        text => [.. System.Text.Encoding.UTF8.GetBytes(text).Select(static b => (int)b)],
        ids => string.Concat(ids.Select(static i => i == 7 ? "�" : ((char)(33 + i % 90)).ToString())));

    [Theory]
    [InlineData("runA", "You are Haru.", true, false, 0f)]
    [InlineData("runB", "", false, false, 0f)]
    [InlineData("runC", "Haru speaks first.", true, true, 2f)]
    public void Conversation_MatchesUpstream(string run, string card, bool withReference, bool speakFirst, float eopBias)
    {
        using HaruReferenceModel model = new(Path.Combine(Dir, "haru_tiny.safetensors"), Path.Combine(Dir, "config.json"), capacity: 256);
        string[] wantText = JsonSerializer.Deserialize<Dictionary<string, string[]>>(File.ReadAllText(Path.Combine(Dir, "haru_session_expected.json")))![run];
        (bool tokens, bool text, float audio, int leftover, int[] ids) = model.ReplayConversation(Path.Combine(Dir, "haru_session_expected.safetensors"),
            run, card, withReference ? "in.reference" : null, wantText, ByteCodec,
            new HaruDuplexOptions { Temperature = 0f, Seed = 1, SpeakFirst = speakFirst, EopBias = eopBias });
        output.WriteLine($"{run}: tokens {string.Join(' ', ids)}; audio rel={audio:E3}");
        Assert.True(tokens, "tokens differ from upstream");
        Assert.True(text, "text differs from upstream");
        Assert.Equal(0, leftover);
        Assert.True(audio < 5e-4f, $"audio relative error {audio:E3}");
    }

    [Fact]
    public void PrepareReference_MatchesUpstream()
    {
        using SafeTensorsLoader expected = new();
        expected.Load(Path.Combine(Dir, "haru_session_expected.safetensors"));
        float[] clip = Read(expected.GetTensor("prep.in"));
        Assert.Equal(Read(expected.GetTensor("prep.out")), HaruAudio.PrepareReference(clip, 24000, maxSeconds: 1.0));
        Assert.Equal(Read(expected.GetTensor("prep.full")), HaruAudio.PrepareReference(clip, 24000));
        Assert.Empty(HaruAudio.PrepareReference(new float[24000], 24000));
        Assert.Equal(100, HaruAudio.PrepareReference(new float[100], 24000).Length);
    }

    [Fact]
    public void Step_RefusesAFrameOfTheWrongLengthAndAStepBeforeStart()
    {
        using SafeTensorsLoader weights = new();
        weights.Load(Path.Combine(Dir, "haru_tiny.safetensors"));
        HaruConfig config = HaruConfig.FromJson(File.ReadAllText(Path.Combine(Dir, "config.json")));
        Dictionary<string, Tensor> tensors = weights.GetAllTensors();
        using CpuBackend backend = new();
        using HaruLm lm = new(config, tensors, capacity: 64);
        using HaruMeanFlowHead head = new(config.Head);
        head.LoadWeights(tensors);
        using HaruCodecSession codec = new(HaruCodec.Create(config.Codec, tensors), backend, config.LatentDim, config.Codec.FixStd);
        HaruDuplexSession session = new(config, lm, head, codec, ByteCodec, backend, new HaruDuplexOptions { Temperature = 0f });
        Assert.Throws<InvalidOperationException>(() => session.Step(new float[config.FrameSamples]));
        session.Start("x", []);
        Assert.Throws<ArgumentException>(() => session.Step(new float[100]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new HaruDuplexSession(config, lm, head, codec, ByteCodec, backend,
            new HaruDuplexOptions { GuidanceScale = 3f }));
    }

    [Fact]
    public void Start_RefusesAPromptThatDoesNotFitTheCapacity_AndStepStopsAfterTheLastFrame()
    {
        using SafeTensorsLoader weights = new();
        weights.Load(Path.Combine(Dir, "haru_tiny.safetensors"));
        HaruConfig config = HaruConfig.FromJson(File.ReadAllText(Path.Combine(Dir, "config.json")));
        Dictionary<string, Tensor> tensors = weights.GetAllTensors();
        using CpuBackend backend = new();
        using HaruMeanFlowHead head = new(config.Head);
        head.LoadWeights(tensors);
        using HaruCodecSession codec = new(HaruCodec.Create(config.Codec, tensors), backend, config.LatentDim, config.Codec.FixStd);
        HaruDuplexOptions options = new() { Temperature = 0f, MaxFrames = 3 };

        using (HaruLm small = new(config, tensors, capacity: 8))
        {
            HaruDuplexSession tooSmall = new(config, small, head, codec, ByteCodec, backend, options);
            Assert.Throws<ArgumentException>(() => tooSmall.Start("a card that does not fit in eight positions", []));
            Assert.Throws<InvalidOperationException>(() => tooSmall.Step(new float[config.FrameSamples])); // the failed Start left no conversation
        }

        using HaruLm lm = new(config, tensors, capacity: 64);
        HaruDuplexSession session = new(config, lm, head, codec, ByteCodec, backend, options);
        Assert.False(session.Start("x", []).Done);
        Assert.False(session.Step(new float[config.FrameSamples]).Done);
        Assert.True(session.Step(new float[config.FrameSamples]).Done);
        Assert.Throws<InvalidOperationException>(() => session.Step(new float[config.FrameSamples]));
    }

    [Fact]
    public void Respond_PlaysARecordingAndReturnsAReplyOfTheSameLength()
    {
        using SafeTensorsLoader weights = new();
        weights.Load(Path.Combine(Dir, "haru_tiny.safetensors"));
        using SafeTensorsLoader expected = new();
        expected.Load(Path.Combine(Dir, "haru_session_expected.safetensors"));
        HaruConfig config = HaruConfig.FromJson(File.ReadAllText(Path.Combine(Dir, "config.json")));
        Dictionary<string, Tensor> tensors = weights.GetAllTensors();
        using CpuBackend backend = new();
        using HaruLm lm = new(config, tensors, capacity: 256);
        using HaruMeanFlowHead head = new(config.Head);
        head.LoadWeights(tensors);
        using HaruCodecSession codec = new(HaruCodec.Create(config.Codec, tensors), backend, config.LatentDim, config.Codec.FixStd);
        HaruDuplexSession session = new(config, lm, head, codec, ByteCodec, backend, new HaruDuplexOptions { Temperature = 0f, Seed = 3 });

        float[] user = Read(expected.GetTensor("in.user"))[..(2 * config.FrameSamples + 500)];
        HaruReply reply = HaruConversation.Respond(session, config, "You are Haru.", [], user, tailFrames: 1);
        Assert.Equal(4 * config.FrameSamples, reply.ModelAudio.Length);
        Assert.Equal(4 * config.FrameSamples, reply.UserAudio.Length);
        Assert.Equal(4, reply.Frames.Count);
        Assert.All(reply.ModelAudio, static s => Assert.InRange(s, -1f, 1f));

        // The same seed repeats the conversation.
        HaruDuplexSession again = new(config, lm, head, codec, ByteCodec, backend, new HaruDuplexOptions { Temperature = 0f, Seed = 3 });
        HaruReply twin = HaruConversation.Respond(again, config, "You are Haru.", [], user, tailFrames: 1);
        Assert.Equal(reply.ModelAudio, twin.ModelAudio);
        Assert.Equal(reply.Text, twin.Text);
    }
}
